//! Saga 32 step 005: attention-tape helpers extracted from
//! `model_tape.rs` so the parent module stays under the
//! sw-checklist function-count budget.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

use crate::error::EvalError;

/// Bundle of attention parameter names + model dimension so the
/// tape helper stays inside the 7-arg clippy budget without needing
/// `#[allow(clippy::too_many_arguments)]`.
pub(crate) struct AttentionInputs<'a> {
    pub(crate) wq: &'a str,
    pub(crate) wk: &'a str,
    pub(crate) wv: &'a str,
    pub(crate) wo: &'a str,
    pub(crate) d_model: usize,
    pub(crate) heads: usize,
    pub(crate) causal: bool,
}

/// Scaled-dot-product attention on the tape. Saga 29 step 013:
/// supports both single-head (`heads = 1`) and multi-head
/// (`heads > 1`) with `d_model % heads == 0`. Rank-2 `[T, d]` and
/// rank-3 `[B, T, d]` inputs both flow through here -- the rank-3
/// path stacks the per-batch outputs back with `Tensor::stack` to
/// avoid the O(B^2) concat chain that the earlier rank-3 lowering
/// produced.
pub(crate) fn attention_tape(
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

pub(crate) fn attention_tape_rank2(
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

pub(crate) fn attention_tape_rank3(
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
