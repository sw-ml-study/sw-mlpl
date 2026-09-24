//! Saga 33 step 004: per-variant `apply_*` helpers for the
//! composite + shape-preserving `ModelSpec` variants. `Chain`
//! and `Residual` recurse through `apply_model`; `RmsNorm` runs
//! the parameter-free per-row RMS normalization in pure Rust.

use mlpl_array::DenseArray;

use crate::model_apply::apply_model;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// `apply(Chain([a, b, ...]), x)` = `apply(b, apply(a, x))`.
/// An `Engram` child additionally receives the chain's ORIGINAL
/// input `x` as its token ids (saga E3 step 1).
pub fn apply_chain(
    children: &[ModelSpec],
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let mut cur = x.clone();
    for child in children {
        cur = match child {
            ModelSpec::Engram { .. } => {
                crate::model_apply_engram_chain::apply_engram_in_chain(child, &cur, x, env)?
            }
            _ => apply_model(child, &cur, env)?,
        };
    }
    Ok(cur)
}

/// `apply(Residual(inner), x)` = `x + apply(inner, x)`. The inner
/// block must preserve input shape, or a shape-mismatch
/// `EvalError::Unsupported` is raised.
pub fn apply_residual(
    inner: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let inner_out = apply_model(inner, x, env)?;
    if inner_out.shape() != x.shape() {
        return Err(EvalError::Unsupported(
            "residual: inner block must preserve input shape".into(),
        ));
    }
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "add", vec![x.clone(), inner_out])
}

/// RMS normalization over the last axis, any rank >= 2 (`[rows, cols]`
/// or `[B, T, d]`): `y[.., :] = x[.., :] / sqrt(mean(x[.., :]^2) + eps)`.
pub fn apply_rms_norm(x: &DenseArray, eps: f64) -> Result<DenseArray, EvalError> {
    let dims = x.shape().dims();
    if dims.len() < 2 {
        return Err(EvalError::Unsupported(
            "rms_norm: input must have rank >= 2 ([rows, cols] or [B, T, d])".into(),
        ));
    }
    let cols = dims[dims.len() - 1].max(1);
    let out: Vec<f64> = x
        .data()
        .chunks(cols)
        .flat_map(|row| {
            let mean_sq = row.iter().map(|v| v * v).sum::<f64>() / cols as f64;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            row.iter().map(move |v| v * scale)
        })
        .collect();
    Ok(DenseArray::new(x.shape().clone(), out)?)
}
