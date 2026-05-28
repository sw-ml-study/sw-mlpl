//! PCA builtins: `pca(X, k)`, `pca_components(X, k)`,
//! `pca_variance_explained(X, k)`.
//!
//! - `pca`: original Saga 16.5 entrypoint. Returns the
//!   centered-and-projected data `[N, k]`.
//! - `pca_components`: saga 33 step 031 (dim-reduction
//!   milestone Phase 1a). Returns the `[k, D]` loadings
//!   matrix -- row `i` is the i-th principal component
//!   direction in original feature space. Feeds the
//!   critical-dimensions heatmap viz (Phase 1b).
//! - `pca_variance_explained`: returns a `[k]` vector of
//!   variance-explained ratios = lambda_i / trace(Cov).
//!   Sums to 1.0 when k == D, less when k < D.
//!
//! Pure compute lifted to `pca_compute.rs` so this module
//! stays under the 7-fn ceiling.

use mlpl_array::{DenseArray, Shape};

use crate::pca_compute::{center_data, compute_cov, extract_components_with_eigenvalues};
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &["pca", "pca_components", "pca_variance_explained"];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "pca" => Some(builtin_pca(args)),
        "pca_components" => Some(builtin_pca_components(args)),
        "pca_variance_explained" => Some(builtin_pca_variance_explained(args)),
        _ => None,
    }
}

/// `pca(X, k) -> Y [N, k]`. Centered-and-projected data.
fn builtin_pca(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate_pca_args("pca", args)?;
    let xc = center_data(&xs, n, d);
    let cov = compute_cov(&xc, n, d);
    let (v, _lambdas) = extract_components_with_eigenvalues(cov, d, k);
    let mut y = vec![0.0_f64; n * k];
    for i in 0..n {
        for c in 0..k {
            let mut s = 0.0_f64;
            for j in 0..d {
                s += xc[i * d + j] * v[c * d + j];
            }
            y[i * k + c] = s;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n, k]), y)?)
}

/// `pca_components(X, k) -> V [k, D]`. Loadings matrix: row
/// `i` is the i-th principal-component direction in
/// original feature space.
fn builtin_pca_components(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate_pca_args("pca_components", args)?;
    let xc = center_data(&xs, n, d);
    let cov = compute_cov(&xc, n, d);
    let (v, _lambdas) = extract_components_with_eigenvalues(cov, d, k);
    Ok(DenseArray::new(Shape::new(vec![k, d]), v)?)
}

/// `pca_variance_explained(X, k) -> [k]`. Per-component
/// variance-explained ratio = lambda_i / trace(Cov). Sums
/// to 1.0 when k == D.
fn builtin_pca_variance_explained(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate_pca_args("pca_variance_explained", args)?;
    let xc = center_data(&xs, n, d);
    let cov = compute_cov(&xc, n, d);
    let total_var: f64 = (0..d).map(|i| cov[i * d + i]).sum();
    let (_v, lambdas) = extract_components_with_eigenvalues(cov, d, k);
    let ratios: Vec<f64> = if total_var > 0.0 {
        lambdas.iter().map(|l| l / total_var).collect()
    } else {
        vec![0.0; k]
    };
    Ok(DenseArray::new(Shape::vector(k), ratios)?)
}

fn validate_pca_args(
    func: &str,
    args: Vec<DenseArray>,
) -> Result<(usize, usize, usize, Vec<f64>), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: func.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let bad = |reason: String| RuntimeError::InvalidArgument {
        func: func.into(),
        reason,
    };
    let x = &args[0];
    if x.rank() != 2 {
        return Err(bad(format!(
            "X must be rank-2 [N, D], got rank {}",
            x.rank()
        )));
    }
    if args[1].rank() != 0 {
        return Err(bad("k must be a scalar".into()));
    }
    let k_f = args[1].data()[0];
    if !k_f.is_finite() || k_f <= 0.0 || k_f.fract() != 0.0 {
        return Err(bad(format!("k must be a positive integer, got {k_f}")));
    }
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    let k = k_f as usize;
    if k > d {
        return Err(bad(format!("k = {k} must be <= D = {d}")));
    }
    let xs = x.data().to_vec();
    if !xs.iter().all(|v| v.is_finite()) {
        return Err(bad("X must contain only finite values (no NaN/Inf)".into()));
    }
    Ok((n, d, k, xs))
}
