//! Incremental single-position forward pass over a KV cache
//! (docs/kv-cache-design.md). Every layer except attention is
//! row-local, so it delegates to the ordinary `apply_model` on
//! the one-row input; causal attention projects ONE row, appends
//! it to the layer's cache, and attends the single query against
//! the cached K/V -- the same dot products in the same order as
//! the full recompute, so CPU outputs are bit-identical.

use mlpl_array::{DenseArray, Shape};

use crate::model_apply::apply_model;
use crate::model_apply_attention::slice_cols;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_core::{AttnKv, GenState};
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Feed ONE token id through the model, appending K/V rows and
/// replacing the state's pending logits with the new output row.
pub fn feed_state_row(gs: &mut GenState, id: f64, env: &Environment) -> Result<(), EvalError> {
    let ids = DenseArray::from_vec(vec![id]);
    let model = gs.model.clone();
    let mut next = 0usize;
    let out = forward_row(&model, &ids, &mut gs.caches, &mut next, env)?;
    gs.logits = out.data().to_vec();
    gs.tokens += 1;
    Ok(())
}

fn forward_row(
    model: &ModelSpec,
    x: &DenseArray,
    caches: &mut [AttnKv],
    next: &mut usize,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    match model {
        ModelSpec::Chain(children) => chain_row(children, x, caches, next, env),
        ModelSpec::Residual(inner) => {
            let inner_out = forward_row(inner, x, caches, next, env)?;
            if inner_out.shape() != x.shape() {
                return Err(EvalError::Unsupported(
                    "residual: inner block must preserve input shape".into(),
                ));
            }
            mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "add", vec![x.clone(), inner_out])
        }
        ModelSpec::Attention { causal: false, .. } => Err(non_causal()),
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            d_model,
            heads,
            causal: true,
        } => {
            let kv = &mut caches[*next];
            *next += 1;
            cached_attention(x, (wq, wk, wv, wo), *d_model, *heads, kv, env)
        }
        row_local => apply_model(row_local, x, env),
    }
}

/// Non-causal attention cannot be cached: position t's output
/// depends on future positions, so incremental generation
/// cannot be exact.
fn non_causal() -> EvalError {
    EvalError::Unsupported(
        "gen_state: non-causal attention cannot be cached -- position t's output \
         depends on future positions, so incremental generation cannot be exact. \
         Use causal_attention in generation chains."
            .into(),
    )
}

/// The chain fold: same shape as `apply_chain`, with an Engram
/// child receiving the chain's ORIGINAL input as its ids.
fn chain_row(
    children: &[ModelSpec],
    x: &DenseArray,
    caches: &mut [AttnKv],
    next: &mut usize,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let mut cur = x.clone();
    for child in children {
        cur = match child {
            ModelSpec::Engram { .. } => {
                crate::model_apply_engram_chain::apply_engram_in_chain(child, &cur, x, env)?
            }
            _ => forward_row(child, &cur, caches, next, env)?,
        };
    }
    Ok(cur)
}

/// One query row against the cached K/V: project the new row
/// with the LIVE weights, append it, attend per head. The last
/// causal row is never masked, so the mask is a no-op here by
/// construction; scale, softmax, and the V blend match the
/// uncached path's per-row arithmetic exactly.
fn cached_attention(
    x: &DenseArray,
    (wq, wk, wv, wo): (&str, &str, &str, &str),
    d_model: usize,
    heads: usize,
    kv: &mut AttnKv,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    use crate::env_api::EnvVars;
    let lookup = |n: &str| -> Result<DenseArray, EvalError> {
        env.get(n)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(n.into()))
    };
    let project = |w: DenseArray| {
        mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![x.clone(), w])
    };
    let q = project(lookup(wq)?)?;
    kv.d_model = d_model;
    kv.k.extend_from_slice(project(lookup(wk)?)?.data());
    kv.v.extend_from_slice(project(lookup(wv)?)?.data());
    kv.rows += 1;
    let t = kv.rows;
    let k_full = DenseArray::new(Shape::new(vec![t, d_model]), kv.k.clone())?;
    let v_full = DenseArray::new(Shape::new(vec![t, d_model]), kv.v.clone())?;
    let d_k = d_model / heads;
    let scale = 1.0 / (d_k as f64).sqrt();
    let mut concat = vec![0.0_f64; d_model];
    for h in 0..heads {
        let head = attend_head(&q, &k_full, &v_full, (h, d_k, scale), env)?;
        concat[h * d_k..(h + 1) * d_k].copy_from_slice(head.data());
    }
    let concat = DenseArray::new(Shape::new(vec![1, d_model]), concat)?;
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![concat, lookup(wo)?])
}

fn attend_head(
    q: &DenseArray,
    k_full: &DenseArray,
    v_full: &DenseArray,
    (h, d_k, scale): (usize, usize, f64),
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let q_h = slice_cols(q, h * d_k, d_k)?;
    let k_h = slice_cols(k_full, h * d_k, d_k)?;
    let v_h = slice_cols(v_full, h * d_k, d_k)?;
    let kt = mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "transpose", vec![k_h])?;
    let scores = mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![q_h, kt])?;
    let t = scores.data().len();
    let scaled: Vec<f64> = scores.data().iter().map(|s| s * scale).collect();
    let scaled = DenseArray::new(Shape::new(vec![1, t]), scaled)?;
    let attn = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "softmax",
        vec![scaled, DenseArray::from_scalar(1.0)],
    )?;
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![attn, v_h])
}
