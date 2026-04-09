//! Built-in dispatch for the Saga 11 model DSL.
//!
//! `linear(in_dim, out_dim, seed)` creates a fresh `ModelSpec::Linear`
//! whose `W` and `b` parameters are stored in the environment under
//! generated names. `apply(model_ident, X)` looks up the model and
//! evaluates it on the given input.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::{ActKind, ModelSpec};
use crate::value::Value;

/// `linear(in_dim, out_dim, seed)`.
pub(crate) fn eval_linear(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "linear".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let in_dim = scalar_usize(&args[0], env, "linear")?;
    let out_dim = scalar_usize(&args[1], env, "linear")?;
    let seed = scalar_f64(&args[2], env, "linear")?;

    let id = env.next_model_id;
    env.next_model_id += 1;
    let w_name = format!("__linear_W_{id}");
    let b_name = format!("__linear_b_{id}");

    // W <- randn(seed, [in_dim, out_dim]) * 0.5 (Xavier-ish small).
    let w_init = mlpl_runtime::call_builtin(
        "randn",
        vec![
            DenseArray::from_scalar(seed),
            DenseArray::new(Shape::new(vec![2]), vec![in_dim as f64, out_dim as f64])?,
        ],
    )?;
    let w_data: Vec<f64> = w_init.data().iter().map(|v| v * 0.5).collect();
    let w = DenseArray::new(Shape::new(vec![in_dim, out_dim]), w_data)?;
    env.set_param(w_name.clone(), w);

    let b = DenseArray::zeros(Shape::new(vec![1, out_dim]));
    env.set_param(b_name.clone(), b);

    Ok(ModelSpec::Linear {
        w: w_name,
        b: b_name,
    })
}

/// `chain(layer_a, layer_b, ...)`. Each argument must evaluate to a
/// `Value::Model`.
pub(crate) fn eval_chain(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    let mut children = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        match crate::eval::eval_expr(arg, env, &mut None)? {
            Value::Model(m) => children.push(m),
            _ => {
                return Err(EvalError::Unsupported(format!(
                    "chain: argument {i} did not evaluate to a model"
                )));
            }
        }
    }
    Ok(ModelSpec::Chain(children))
}

/// `residual(inner_model)`. Wraps a single model argument in a
/// skip-connection node.
pub(crate) fn eval_residual(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "residual".into(),
            expected: 1,
            got: args.len(),
        });
    }
    match crate::eval::eval_expr(&args[0], env, &mut None)? {
        Value::Model(m) => Ok(ModelSpec::Residual(Box::new(m))),
        _ => Err(EvalError::Unsupported(
            "residual: argument must evaluate to a model".into(),
        )),
    }
}

/// `attention(d_model, heads, seed)` -- multi-head self-attention.
/// Allocates four `[d_model, d_model]` weight params and registers
/// them as trainable in the env. Names are namespaced by the env's
/// `next_model_id` so multiple attention layers do not collide.
pub(crate) fn eval_attention(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "attention".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let d_model = scalar_usize(&args[0], env, "attention")?;
    let heads = scalar_usize(&args[1], env, "attention")?;
    let seed = scalar_f64(&args[2], env, "attention")?;
    if heads == 0 || d_model % heads != 0 {
        return Err(EvalError::Unsupported(format!(
            "attention: d_model ({d_model}) must be divisible by heads ({heads})"
        )));
    }
    let id = env.next_model_id;
    env.next_model_id += 1;
    let wq = format!("__attn_Wq_{id}");
    let wk = format!("__attn_Wk_{id}");
    let wv = format!("__attn_Wv_{id}");
    let wo = format!("__attn_Wo_{id}");
    // Use a small offset on the seed so the four projections do not
    // start out identical.
    for (i, name) in [&wq, &wk, &wv, &wo].iter().enumerate() {
        let init = mlpl_runtime::call_builtin(
            "randn",
            vec![
                DenseArray::from_scalar(seed + i as f64),
                DenseArray::new(Shape::new(vec![2]), vec![d_model as f64, d_model as f64])?,
            ],
        )?;
        let scaled: Vec<f64> = init.data().iter().map(|v| v * 0.5).collect();
        let arr = DenseArray::new(Shape::new(vec![d_model, d_model]), scaled)?;
        env.set_param((*name).clone(), arr);
    }
    Ok(ModelSpec::Attention {
        wq,
        wk,
        wv,
        wo,
        d_model,
        heads,
    })
}

/// `rms_norm(dim)` -- parameter-free per-row RMS normalization.
pub(crate) fn eval_rms_norm(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "rms_norm".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let dim = scalar_usize(&args[0], env, "rms_norm")?;
    Ok(ModelSpec::RmsNorm { dim })
}

/// Parameter-free activation layer constructors. Returns the
/// matching `ActKind` if `name` is recognized.
#[must_use]
pub(crate) fn activation_kind(name: &str) -> Option<ActKind> {
    Some(match name {
        "tanh_layer" => ActKind::Tanh,
        "relu_layer" => ActKind::Relu,
        "softmax_layer" => ActKind::Softmax,
        _ => return None,
    })
}

/// `apply(model_ident, X)`.
pub(crate) fn eval_apply(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "apply".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "apply: first argument must be a model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let x = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    apply_model(&model, &x, env)
}

fn apply_model(
    model: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    match model {
        ModelSpec::Linear { w, b } => {
            let w_arr = env
                .get(w)
                .ok_or_else(|| EvalError::UndefinedVariable(w.clone()))?;
            let b_arr = env
                .get(b)
                .ok_or_else(|| EvalError::UndefinedVariable(b.clone()))?;
            let xw = x.matmul(w_arr)?;
            // Broadcast b ([1, out]) up to [n, out] via ones([n, 1]) @ b.
            let n = xw.shape().dims()[0];
            let ones = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
            let b_broadcast = ones.matmul(b_arr)?;
            Ok(xw.apply_binop(&b_broadcast, |a, c| a + c)?)
        }
        ModelSpec::Chain(children) => {
            let mut cur = x.clone();
            for child in children {
                cur = apply_model(child, &cur, env)?;
            }
            Ok(cur)
        }
        ModelSpec::Activation(kind) => match kind {
            ActKind::Tanh => Ok(x.map(f64::tanh)),
            ActKind::Relu => Ok(x.map(|v| if v > 0.0 { v } else { 0.0 })),
            ActKind::Softmax => Ok(mlpl_runtime::call_builtin(
                "softmax",
                vec![x.clone(), DenseArray::from_scalar(1.0)],
            )?),
        },
        ModelSpec::Residual(inner) => {
            let inner_out = apply_model(inner, x, env)?;
            if inner_out.shape() != x.shape() {
                return Err(EvalError::Unsupported(
                    "residual: inner block must preserve input shape".into(),
                ));
            }
            Ok(x.apply_binop(&inner_out, |a, b| a + b)?)
        }
        ModelSpec::RmsNorm { .. } => apply_rms_norm(x),
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            d_model,
            heads,
        } => apply_attention(x, wq, wk, wv, wo, *d_model, *heads, env),
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_attention(
    x: &DenseArray,
    wq: &str,
    wk: &str,
    wv: &str,
    wo: &str,
    d_model: usize,
    heads: usize,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    if dims.len() != 2 || dims[1] != d_model {
        return Err(EvalError::Unsupported(format!(
            "attention: input must be [seq, {d_model}], got {:?}",
            dims
        )));
    }
    let seq = dims[0];
    let d_k = d_model / heads;

    let wq_a = env
        .get(wq)
        .ok_or_else(|| EvalError::UndefinedVariable(wq.into()))?;
    let wk_a = env
        .get(wk)
        .ok_or_else(|| EvalError::UndefinedVariable(wk.into()))?;
    let wv_a = env
        .get(wv)
        .ok_or_else(|| EvalError::UndefinedVariable(wv.into()))?;
    let wo_a = env
        .get(wo)
        .ok_or_else(|| EvalError::UndefinedVariable(wo.into()))?;

    let q = x.matmul(wq_a)?;
    let k = x.matmul(wk_a)?;
    let v = x.matmul(wv_a)?;

    let scale = 1.0 / (d_k as f64).sqrt();
    // Per-head attention. Concatenate per-head outputs back into a
    // [seq, d_model] matrix in the same column layout as the input.
    let mut concat = vec![0.0_f64; seq * d_model];
    for h in 0..heads {
        let q_h = slice_cols(&q, h * d_k, d_k)?;
        let k_h = slice_cols(&k, h * d_k, d_k)?;
        let v_h = slice_cols(&v, h * d_k, d_k)?;
        let scores = q_h.matmul(&k_h.transpose())?;
        let scaled: Vec<f64> = scores.data().iter().map(|s| s * scale).collect();
        let scores_scaled = DenseArray::new(Shape::new(vec![seq, seq]), scaled)?;
        let attn = mlpl_runtime::call_builtin(
            "softmax",
            vec![scores_scaled, DenseArray::from_scalar(1.0)],
        )?;
        let head_out = attn.matmul(&v_h)?; // [seq, d_k]
        for r in 0..seq {
            for c in 0..d_k {
                concat[r * d_model + h * d_k + c] = head_out.data()[r * d_k + c];
            }
        }
    }
    let concat = DenseArray::new(Shape::new(vec![seq, d_model]), concat)?;
    Ok(concat.matmul(wo_a)?)
}

/// Extract `width` consecutive columns starting at `start` from a
/// rank-2 matrix.
fn slice_cols(x: &DenseArray, start: usize, width: usize) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    let rows = dims[0];
    let cols = dims[1];
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        for c in 0..width {
            out.push(x.data()[r * cols + start + c]);
        }
    }
    Ok(DenseArray::new(Shape::new(vec![rows, width]), out)?)
}

/// Tape-side analogue of `apply_model` used by `grad(loss_with_apply, ...)`
/// and therefore by `adam(loss_with_apply, mdl, ...)`. Walks the model
/// and emits the same primitive ops onto the autograd tape so that
/// gradients flow back into each layer's trainable leaves. Layers
/// without primitive tape equivalents (residual/rms_norm/attention)
/// are reported as unsupported for now; lowering them is a follow-up
/// needed before the transformer-block demo can train end-to-end.
pub(crate) fn apply_model_tape(
    model: &ModelSpec,
    x: Tensor,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    match model {
        ModelSpec::Linear { w, b } => {
            let w_t = params
                .get(w)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(w.clone()))?;
            let b_t = params
                .get(b)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(b.clone()))?;
            let xw = x.matmul(&w_t);
            // Match apply_model's eager broadcast recipe: b is [1, out],
            // lift it to [n, out] via ones([n, 1]) @ b so the tape graph
            // is identical to the hand-rolled form.
            let n = xw.value().shape().dims()[0];
            let ones_arr = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
            let ones_t = Tensor::leaf(Rc::clone(tape), ones_arr, false);
            let b_broadcast = ones_t.matmul(&b_t);
            Ok(xw.add(&b_broadcast))
        }
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
        ModelSpec::Residual(_) | ModelSpec::RmsNorm { .. } | ModelSpec::Attention { .. } => {
            Err(EvalError::Unsupported(
                "grad: apply() through residual/rms_norm/attention is not yet supported; \
                 inline those layers in the loss expression for now"
                    .into(),
            ))
        }
    }
}

/// Per-row RMS normalization: `y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)`.
fn apply_rms_norm(x: &DenseArray) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(
            "rms_norm: input must be a rank-2 [rows, cols] matrix".into(),
        ));
    }
    let rows = dims[0];
    let cols = dims[1];
    let eps = 1e-8;
    let src = x.data();
    let mut out = Vec::with_capacity(src.len());
    for r in 0..rows {
        let row = &src[r * cols..(r + 1) * cols];
        let mean_sq: f64 = row.iter().map(|v| v * v).sum::<f64>() / cols.max(1) as f64;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for v in row {
            out.push(v * scale);
        }
    }
    Ok(DenseArray::new(Shape::new(vec![rows, cols]), out)?)
}

fn scalar_f64(expr: &Expr, env: &mut Environment, func: &str) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: expected a scalar argument"
        )));
    }
    Ok(arr.data()[0])
}

fn scalar_usize(expr: &Expr, env: &mut Environment, func: &str) -> Result<usize, EvalError> {
    let v = scalar_f64(expr, env, func)?;
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: dimension must be a non-negative integer"
        )));
    }
    Ok(v as usize)
}
