//! Tape-side lowering of the Saga 11 model DSL.
//!
//! This is the autograd twin of `model_dispatch::apply_model`: given
//! a `ModelSpec` and an input `Tensor` handle, it walks the model
//! tree and emits the same primitive ops onto the autograd tape so
//! that `grad(loss_with_apply, ...)` -- and therefore every optimizer
//! built on top of it -- can route gradients back to each layer's
//! trainable parameter leaves.
//!
//! Coverage today:
//!
//! - `linear`, `chain`, activation layers, `residual`, `rms_norm`,
//!   and single- and multi-head `attention` are fully lowered.
//! - Saga 29 step 013: multi-head attention on the tape is
//!   lowered by reshape + take + per-head scaled-dot-product +
//!   `Tensor::stack` over the head axis. Backward flows through
//!   the same primitives. (Initial saga-29 versions chained
//!   binary concats here; step 013 replaced the chain with a
//!   single `stack` tape node.)

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

use crate::error::EvalError;
use mlpl_eval_core::model::{ActKind, ModelSpec};

/// Tape-side analogue of `model_dispatch::apply_model`. Walks the
/// model and emits the matching primitive ops on the autograd tape.
pub(crate) fn apply_model_tape(
    model: &ModelSpec,
    x: Tensor,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    match model {
        ModelSpec::Linear { w, b } => linear_tape(&x, w, b, tape, params),
        ModelSpec::Chain(children) => {
            let mut cur = x;
            for child in children {
                cur = apply_model_tape(child, cur, tape, params)?;
            }
            Ok(cur)
        }
        ModelSpec::Activation(kind) => Ok(match kind {
            ActKind::Tanh => x.tanh(),
            ActKind::Relu => x.relu(),
            ActKind::Softmax => x.softmax(),
        }),
        ModelSpec::Residual(inner) => {
            let inner_out = apply_model_tape(inner, x.clone(), tape, params)?;
            Ok(x.add(&inner_out))
        }
        ModelSpec::RmsNorm { .. } => rms_norm_tape(&x, tape),
        ModelSpec::Embedding { table, vocab, .. } => {
            let table_t = params
                .get(table)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(table.clone()))?;
            // The token id array enters the tape as a non-trainable
            // input leaf; we use its eager value to build a one-hot
            // matrix and matmul against the trainable table tensor.
            // Backprop then routes through matmul straight into the
            // table's gradient buffer.
            let tokens_arr = x.value().clone();
            let onehot_arr = onehot_from_tokens(&tokens_arr, *vocab)?;
            let onehot_t = Tensor::leaf(Rc::clone(tape), onehot_arr, false);
            Ok(onehot_t.matmul(&table_t))
        }
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            d_model,
            heads,
            causal,
        } => {
            let inputs = AttentionInputs {
                wq,
                wk,
                wv,
                wo,
                d_model: *d_model,
                heads: *heads,
                causal: *causal,
            };
            attention_tape(&x, &inputs, tape, params)
        }
        ModelSpec::LinearLora {
            w,
            b,
            a,
            b_adapter,
            rank,
            alpha,
            ..
        } => linear_lora_tape(
            &x,
            &LinearLoraTapeInputs {
                w,
                b,
                a,
                b_adapter,
                rank: *rank,
                alpha: *alpha,
            },
            tape,
            params,
        ),
    }
}

/// Named-field inputs for one `linear_lora_tape` call so
/// the helper stays at 4 args and does not trip
/// `clippy::too_many_arguments`.
struct LinearLoraTapeInputs<'a> {
    w: &'a str,
    b: &'a str,
    a: &'a str,
    b_adapter: &'a str,
    rank: usize,
    alpha: f64,
}

fn linear_lora_tape(
    x: &Tensor,
    inputs: &LinearLoraTapeInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let w_t = params
        .get(inputs.w)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.w.into()))?;
    let b_t = params
        .get(inputs.b)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.b.into()))?;
    let a_t = params
        .get(inputs.a)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.a.into()))?;
    let b_adapt_t = params
        .get(inputs.b_adapter)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.b_adapter.into()))?;
    // Base: X @ W
    let xw = x.matmul(&w_t);
    // Adapter: (alpha / rank) * X @ A @ B
    let xa = x.matmul(&a_t);
    let xab = xa.matmul(&b_adapt_t);
    let scale = inputs.alpha / inputs.rank as f64;
    let scale_t = Tensor::leaf(Rc::clone(tape), DenseArray::from_scalar(scale), false);
    let xab_scaled = xab.mul(&scale_t);
    // Bias: broadcast [1, out] -> [n, out] via ones([n, 1]) @ b.
    let n = xw.value().shape().dims()[0];
    let ones_arr = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let ones_t = Tensor::leaf(Rc::clone(tape), ones_arr, false);
    Ok(xw.add(&xab_scaled).add(&ones_t.matmul(&b_t)))
}

fn linear_tape(
    x: &Tensor,
    w: &str,
    b: &str,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let w_t = params
        .get(w)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(w.into()))?;
    let b_t = params
        .get(b)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(b.into()))?;
    let xw = x.matmul(&w_t);
    // b is [1, out]; lift to [n, out] via ones([n, 1]) @ b so the
    // tape graph matches the hand-rolled broadcast form exactly.
    let n = xw.value().shape().dims()[0];
    let ones_arr = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let ones_t = Tensor::leaf(Rc::clone(tape), ones_arr, false);
    Ok(xw.add(&ones_t.matmul(&b_t)))
}

/// Per-row RMS normalization on the tape:
/// `y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)`.
/// `sqrt(v)` is encoded as `exp(-0.5 * log(v))` because the tape
/// does not yet expose a direct sqrt op. Per-row mean is encoded as
/// `matmul(x, ones([cols, 1])) / cols`, and the broadcast back to
/// `[rows, cols]` as `rsqrt @ ones([1, cols])`.
fn rms_norm_tape(x: &Tensor, tape: &Rc<Tape>) -> Result<Tensor, EvalError> {
    let dims = x.value().shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(
            "rms_norm: input must be a rank-2 [rows, cols] matrix".into(),
        ));
    }
    let cols = dims[1];
    let eps = 1e-8_f64;
    let leaf = |v: DenseArray| Tensor::leaf(Rc::clone(tape), v, false);
    let ones_col = leaf(DenseArray::new(Shape::new(vec![cols, 1]), vec![1.0; cols])?);
    let ones_row = leaf(DenseArray::new(Shape::new(vec![1, cols]), vec![1.0; cols])?);
    let inv_cols = leaf(DenseArray::from_scalar(1.0 / cols as f64));
    let eps_t = leaf(DenseArray::from_scalar(eps));
    let half_neg = leaf(DenseArray::from_scalar(-0.5));
    let row_mean_eps = x.mul(x).matmul(&ones_col).mul(&inv_cols).add(&eps_t);
    let rsqrt = row_mean_eps.log().mul(&half_neg).exp();
    Ok(x.mul(&rsqrt.matmul(&ones_row)))
}

/// One-hot encode a 1-D token id array `[N]` into `[N, vocab]`. Mirrors
/// the eager helper in `model_dispatch` so the tape sees an identical
/// matrix; consolidating the two would couple the modules more tightly
/// than this small allocation is worth.
fn onehot_from_tokens(tokens: &DenseArray, vocab: usize) -> Result<DenseArray, EvalError> {
    let dims = tokens.shape().dims();
    if dims.len() != 1 {
        let msg = format!("embed (tape): tokens must be a 1-D [N] array, got shape {dims:?}");
        return Err(EvalError::Unsupported(msg));
    }
    let n = dims[0];
    let mut data = vec![0.0_f64; n * vocab];
    for (row, &id_f) in tokens.data().iter().enumerate() {
        let id = validate_token_id(row, id_f, vocab)?;
        data[row * vocab + id] = 1.0;
    }
    DenseArray::new(Shape::new(vec![n, vocab]), data).map_err(|e| {
        EvalError::Unsupported(format!("embed (tape): one-hot construction failed: {e}"))
    })
}

fn validate_token_id(row: usize, id_f: f64, vocab: usize) -> Result<usize, EvalError> {
    if !id_f.is_finite() || id_f < 0.0 || id_f.fract() != 0.0 {
        let msg =
            format!("embed (tape): token at position {row} = {id_f} is not a non-negative integer");
        return Err(EvalError::Unsupported(msg));
    }
    let id = id_f as usize;
    if id >= vocab {
        let msg =
            format!("embed (tape): token at position {row} = {id} out of vocab range [0, {vocab})");
        return Err(EvalError::Unsupported(msg));
    }
    Ok(id)
}

/// Bundle of attention parameter names + model dimension so the
/// tape helper stays inside the 7-arg clippy budget without needing
/// `#[allow(clippy::too_many_arguments)]`.
struct AttentionInputs<'a> {
    wq: &'a str,
    wk: &'a str,
    wv: &'a str,
    wo: &'a str,
    d_model: usize,
    heads: usize,
    causal: bool,
}

/// Scaled-dot-product attention on the tape. Saga 29 step 013:
/// supports both single-head (`heads = 1`) and multi-head
/// (`heads > 1`) with `d_model % heads == 0`. Rank-2 `[T, d]` and
/// rank-3 `[B, T, d]` inputs both flow through here -- the rank-3
/// path stacks the per-batch outputs back with `Tensor::stack` to
/// avoid the O(B^2) concat chain that the earlier rank-3 lowering
/// produced.
fn attention_tape(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let dims = x.value().shape().dims().to_vec();
    if dims.len() == 3 && dims[2] == inputs.d_model {
        return attention_tape_rank3(x, inputs, tape, params, dims);
    }
    if dims.len() != 2 || dims[1] != inputs.d_model {
        return Err(EvalError::Unsupported(format!(
            "attention: input must be [seq, {}] or [B, T, {}], got {dims:?}",
            inputs.d_model, inputs.d_model
        )));
    }
    attention_tape_rank2(x, inputs, tape, params)
}

fn attention_tape_rank2(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let fetch = |name: &str| -> Result<Tensor, EvalError> {
        params
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(name.into()))
    };
    let wq_t = fetch(inputs.wq)?;
    let wk_t = fetch(inputs.wk)?;
    let wv_t = fetch(inputs.wv)?;
    let wo_t = fetch(inputs.wo)?;
    let q = x.matmul(&wq_t);
    let k = x.matmul(&wk_t);
    let v = x.matmul(&wv_t);
    let context = if inputs.heads == 1 {
        attention_single_head_context(&q, &k, &v, inputs.d_model, inputs.causal, tape)?
    } else {
        attention_multi_head_context(
            &q,
            &k,
            &v,
            inputs.d_model,
            inputs.heads,
            inputs.causal,
            tape,
        )?
    };
    Ok(context.matmul(&wo_t))
}

/// Single-head context: `softmax(scale * Q @ K^T + mask) @ V`.
/// Output shape matches `Q` (rank-2 `[T, d_model]`).
fn attention_single_head_context(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_model: usize,
    causal: bool,
    tape: &Rc<Tape>,
) -> Result<Tensor, EvalError> {
    let scale = Tensor::leaf(
        Rc::clone(tape),
        DenseArray::from_scalar(1.0 / (d_model as f64).sqrt()),
        false,
    );
    let scores = q.matmul(&k.transpose()).mul(&scale);
    let masked = if causal {
        let seq = scores.value().shape().dims()[0];
        let mask = Tensor::leaf(Rc::clone(tape), causal_mask_array(seq)?, false);
        scores.add(&mask)
    } else {
        scores
    };
    Ok(masked.softmax().matmul(v))
}

/// Multi-head context: split Q/K/V column-wise into `heads` slabs
/// of width `d_k = d_model / heads`, run per-head scaled-dot-
/// product on each, and concatenate the per-head outputs back
/// over the column axis. Output shape `[T, d_model]`.
fn attention_multi_head_context(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_model: usize,
    heads: usize,
    causal: bool,
    tape: &Rc<Tape>,
) -> Result<Tensor, EvalError> {
    if !d_model.is_multiple_of(heads) {
        return Err(EvalError::Unsupported(format!(
            "attention: d_model ({d_model}) must be divisible by heads ({heads})"
        )));
    }
    let d_k = d_model / heads;
    let t = q.value().shape().dims()[0];
    let head_shape = mlpl_array::Shape::new(vec![t, heads, d_k]);
    let q_split = q.reshape(head_shape.clone());
    let k_split = k.reshape(head_shape.clone());
    let v_split = v.reshape(head_shape);
    let scale = Tensor::leaf(
        Rc::clone(tape),
        DenseArray::from_scalar(1.0 / (d_k as f64).sqrt()),
        false,
    );
    let mask = if causal {
        Some(Tensor::leaf(Rc::clone(tape), causal_mask_array(t)?, false))
    } else {
        None
    };
    let mut head_outs: Vec<Tensor> = Vec::with_capacity(heads);
    for h in 0..heads {
        let q_h = q_split.take(1, h);
        let k_h = k_split.take(1, h);
        let v_h = v_split.take(1, h);
        let scores = q_h.matmul(&k_h.transpose()).mul(&scale);
        let masked = if let Some(m) = &mask {
            scores.add(m)
        } else {
            scores
        };
        head_outs.push(masked.softmax().matmul(&v_h));
    }
    Ok(Tensor::stack(&head_outs, 1))
}

fn attention_tape_rank3(
    x: &Tensor,
    inputs: &AttentionInputs<'_>,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
    dims: Vec<usize>,
) -> Result<Tensor, EvalError> {
    let (batch, t, d) = (dims[0], dims[1], dims[2]);
    if batch == 0 {
        return Err(EvalError::Unsupported(
            "attention: rank-3 input had zero batch entries".into(),
        ));
    }
    let mut per_batch: Vec<Tensor> = Vec::with_capacity(batch);
    for b in 0..batch {
        let x_b = x.take(0, b);
        let out_b = attention_tape_rank2(&x_b, inputs, tape, params)?;
        per_batch.push(out_b.reshape(mlpl_array::Shape::new(vec![1, t, d])));
    }
    Ok(Tensor::stack(&per_batch, 0))
}

/// Build a `[seq, seq]` additive causal mask: zero on and below the
/// diagonal, a large negative value strictly above. Adding it to the
/// pre-softmax scores pushes non-causal positions to ~0 probability
/// after softmax, without introducing a new tape primitive.
fn causal_mask_array(seq: usize) -> Result<DenseArray, EvalError> {
    let neg = -1.0e9_f64;
    let mut data = vec![0.0_f64; seq * seq];
    for r in 0..seq {
        for c in (r + 1)..seq {
            data[r * seq + c] = neg;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![seq, seq]), data)?)
}

/// Shape + index validation for `cross_entropy(logits, targets)`.
/// Shared between the forward-only runtime builtin path (via `eval.rs`)
/// and the tape-side path in `grad.rs`, so errors look identical either
/// way. Returns a flat `Vec<usize>` of class indices ready to hand to
/// `Tensor::cross_entropy`.
pub(crate) fn validate_cross_entropy_targets(
    logits: &DenseArray,
    targets: &DenseArray,
) -> Result<Vec<usize>, EvalError> {
    let dims = logits.shape().dims();
    let (n, v) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        r => {
            return Err(EvalError::Unsupported(format!(
                "cross_entropy: logits must be rank 2 or 3, got rank {r}"
            )));
        }
    };
    if targets.elem_count() != n {
        return Err(EvalError::ShapeMismatch {
            op: "cross_entropy".into(),
            expected: mlpl_core::LabeledShape::new(vec![n], vec![None]),
            actual: mlpl_core::LabeledShape::new(
                targets.shape().dims().to_vec(),
                vec![None; targets.rank()],
            ),
        });
    }
    let mut idx = Vec::with_capacity(n);
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 || (t as usize) >= v {
            return Err(EvalError::Unsupported(format!(
                "cross_entropy: target[{i}] = {t} is not a valid class index in 0..{v}"
            )));
        }
        idx.push(t as usize);
    }
    Ok(idx)
}
