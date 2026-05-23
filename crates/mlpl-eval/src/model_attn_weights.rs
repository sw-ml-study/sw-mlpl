//! Saga 33 step 004: read-only attention-weights forward pass
//! extracted from `model_dispatch.rs`. Threads input through
//! the model tree, locates the first `Attention` layer, and
//! returns its softmax weights for visualization. Rank-3 [B, T,
//! d_model] is looped per batch via `batched_attn_weights`.

use mlpl_array::{DenseArray, Shape};

use crate::env::Environment;
use crate::error::EvalError;
use crate::model_apply::apply_model;
use crate::model_apply_attention::slice_cols;
use mlpl_eval_core::model::ModelSpec;

/// Walk the model tree, threading `x` through each layer until
/// we hit the first `Attention` node; then return its softmax
/// weights.
pub(crate) fn extract_attn_weights(
    m: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let not_found =
        || EvalError::Unsupported("attention_weights: no Attention layer found in model".into());
    match m {
        ModelSpec::Attention {
            wq,
            wk,
            d_model,
            heads,
            causal,
            ..
        } => compute_attn_weights(x, wq, wk, *d_model, *heads, *causal, env),
        ModelSpec::Chain(children) => {
            let mut cur = x.clone();
            for child in children {
                if matches!(child, ModelSpec::Attention { .. }) {
                    return extract_attn_weights(child, &cur, env);
                }
                cur = apply_model(child, &cur, env)?;
            }
            Err(not_found())
        }
        ModelSpec::Residual(inner) => extract_attn_weights(inner, x, env),
        _ => Err(not_found()),
    }
}

/// One head's attention weight matrix: `softmax(scale * Q_h @
/// K_h^T)` with optional causal upper-triangle masking. Returns
/// `[seq, seq]` for the caller to concatenate.
fn attn_head_weights(
    q: &DenseArray,
    k: &DenseArray,
    h: usize,
    dims: (usize, usize),
    cfg: (f64, bool),
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let (d_k, seq) = dims;
    let (scale, causal) = cfg;
    let q_h = slice_cols(q, h * d_k, d_k)?;
    let k_h = slice_cols(k, h * d_k, d_k)?;
    let kt = crate::device::dispatched_call(env, "transpose", vec![k_h])?;
    let qk = crate::device::dispatched_call(env, "matmul", vec![q_h, kt])?;
    let scaled: Vec<f64> = qk
        .data()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if causal && i % seq > i / seq {
                -1e9
            } else {
                s * scale
            }
        })
        .collect();
    let scores = DenseArray::new(Shape::new(vec![seq, seq]), scaled)?;
    crate::device::dispatched_call(env, "softmax", vec![scores, DenseArray::from_scalar(1.0)])
}

fn compute_attn_weights(
    x: &DenseArray,
    wq: &str,
    wk: &str,
    d_model: usize,
    heads: usize,
    causal: bool,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    if dims.len() == 3 && dims[2] == d_model {
        return batched_attn_weights(x, wq, wk, d_model, heads, causal, env);
    }
    if dims.len() != 2 || dims[1] != d_model {
        return Err(EvalError::Unsupported(format!(
            "attention_weights: input must be [seq, {d_model}] or [B, T, {d_model}], got {:?}",
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
    let q = crate::device::dispatched_call(env, "matmul", vec![x.clone(), wq_a.clone()])?;
    let k = crate::device::dispatched_call(env, "matmul", vec![x.clone(), wk_a.clone()])?;
    let scale = 1.0 / (d_k as f64).sqrt();
    let mut all = Vec::with_capacity(heads * seq * seq);
    for h in 0..heads {
        let attn = attn_head_weights(&q, &k, h, (d_k, seq), (scale, causal), env)?;
        all.extend_from_slice(attn.data());
    }
    let shape = if heads == 1 {
        vec![seq, seq]
    } else {
        vec![heads, seq, seq]
    };
    Ok(DenseArray::new(Shape::new(shape), all)?)
}

/// Rank-3 [B, T, d_model] attention-weights driver.
fn batched_attn_weights(
    x: &DenseArray,
    wq: &str,
    wk: &str,
    d_model: usize,
    heads: usize,
    causal: bool,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    let (batch, t) = (dims[0], dims[1]);
    let inner_shape: Vec<usize> = if heads == 1 {
        vec![t, t]
    } else {
        vec![heads, t, t]
    };
    let mut data = Vec::with_capacity(batch * inner_shape.iter().product::<usize>());
    for b in 0..batch {
        let x_b = x.take(0, b)?;
        let w_b = compute_attn_weights(&x_b, wq, wk, d_model, heads, causal, env)?;
        data.extend_from_slice(w_b.data());
    }
    let out_shape = std::iter::once(batch).chain(inner_shape).collect();
    Ok(DenseArray::new(Shape::new(out_shape), data)?)
}
