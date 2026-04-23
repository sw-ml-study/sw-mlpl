//! Saga 16 step 002: `tsne` argument validation.
//!
//! Split out of `tsne_builtin.rs` so the orchestrator stays
//! short and every precondition error lives in one place.
//! The public return type `TsneArgs` is the "validated +
//! normalised" shape every subsequent phase reads from.

use mlpl_array::DenseArray;

use crate::error::RuntimeError;

/// Validated + normalised `tsne` call parameters.
pub(crate) struct TsneArgs {
    pub(crate) n: usize,
    pub(crate) d: usize,
    pub(crate) perplexity: f64,
    pub(crate) iters: usize,
    pub(crate) seed: f64,
    pub(crate) xs: Vec<f64>,
}

/// Top-level validation orchestrator. Returns `TsneArgs`
/// on success, a `RuntimeError` surfacing the first
/// precondition violation otherwise.
pub(crate) fn validate_tsne_args(args: Vec<DenseArray>) -> Result<TsneArgs, RuntimeError> {
    if args.len() != 4 {
        return Err(RuntimeError::ArityMismatch {
            func: "tsne".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let (n, d, xs) = validate_x(&args[0])?;
    let perplexity = validate_perplexity(&args[1], n)?;
    let iters = validate_iters(&args[2])?;
    let seed = scalar(&args[3], "seed")?;
    Ok(TsneArgs {
        n,
        d,
        perplexity,
        iters,
        seed,
        xs,
    })
}

/// Rank + finite check on `X`. Returns `(N, D, xs)` where
/// `xs` is the owned row-major flat data.
fn validate_x(x: &DenseArray) -> Result<(usize, usize, Vec<f64>), RuntimeError> {
    if x.rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: "tsne".into(),
            reason: format!("X must be rank-2 [N, D], got rank {}", x.rank()),
        });
    }
    for (i, v) in x.data().iter().enumerate() {
        if !v.is_finite() {
            return Err(RuntimeError::InvalidArgument {
                func: "tsne".into(),
                reason: format!("X must be finite; element {i} is {v}"),
            });
        }
    }
    let dims = x.shape().dims();
    Ok((dims[0], dims[1], x.data().to_vec()))
}

/// Perplexity must be a positive scalar and strictly less
/// than `N` (else the per-row binary search cannot find a
/// beta that matches the target entropy).
fn validate_perplexity(arg: &DenseArray, n: usize) -> Result<f64, RuntimeError> {
    let v = scalar(arg, "perplexity")?;
    if !v.is_finite() || v <= 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "tsne".into(),
            reason: format!("perplexity must be positive, got {v}"),
        });
    }
    if v >= n as f64 {
        return Err(RuntimeError::InvalidArgument {
            func: "tsne".into(),
            reason: format!(
                "perplexity {v} must be < N = {n}; the binary search cannot \
                 find a beta that matches perplexity >= N"
            ),
        });
    }
    Ok(v)
}

/// `iters` must be a positive integer scalar.
fn validate_iters(arg: &DenseArray) -> Result<usize, RuntimeError> {
    let v = scalar(arg, "iters")?;
    if !v.is_finite() || v < 1.0 || v.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "tsne".into(),
            reason: format!("iters must be a positive integer, got {v}"),
        });
    }
    Ok(v as usize)
}

/// Unwrap a rank-0 `DenseArray` into its single scalar
/// value; otherwise surface a `tsne: <name> must be a
/// scalar` error.
pub(crate) fn scalar(a: &DenseArray, name: &str) -> Result<f64, RuntimeError> {
    if a.rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "tsne".into(),
            reason: format!("{name} must be a scalar"),
        });
    }
    Ok(a.data()[0])
}
