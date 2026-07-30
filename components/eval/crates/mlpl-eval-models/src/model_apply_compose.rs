//! Saga 33 step 004: per-variant `apply_*` helpers for the
//! composite + shape-preserving `ModelSpec` variants. `Chain`
//! and `Residual` recurse through `apply_model`; `RmsNorm` runs
//! the parameter-free per-row RMS normalization in pure Rust.

use mlpl_array::{DenseArray, Shape};

use crate::model_apply::apply_model;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// `apply(Chain([a, b, ...]), x)` = `apply(b, apply(a, x))`.
pub fn apply_chain(
    children: &[ModelSpec],
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let mut cur = x.clone();
    for child in children {
        cur = apply_model(child, &cur, env)?;
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

/// Per-row RMS normalization: `y[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + eps)`.
pub fn apply_rms_norm(x: &DenseArray) -> Result<DenseArray, EvalError> {
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
