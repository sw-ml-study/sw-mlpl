//! Saga 16.5 step 001: `pca(X, k)` builtin.
//!
//! Top-k principal component analysis via power
//! iteration + deflation, wrapped into a single builtin.
//! Returns the centered-and-projected data `[N, k]`, not
//! the k components themselves. Callers who need the
//! eigenvectors can run the power-iteration composition
//! pattern directly; see the Saga 8 tutorial lesson.
//!
//! Algorithm: center `Xc = X - col_means(X)`, form
//! `Cov = Xc^T Xc / N`, then for each component power-
//! iterate the dominant eigenvector of `Cov` for
//! `POWER_ITERS = 50` steps, record it, and deflate
//! `Cov -= lambda * (v outer v)`. Stack components into
//! `V [k, D]` and return `Xc @ V^T [N, k]`.
//!
//! See `contracts/eval-contract/pca.md`.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

/// Number of power-iteration steps per component. 50 is
/// comfortably convergent at the matrix sizes we expect
/// (D up to a few hundred) while keeping the algorithm
/// deterministic and cheap.
const POWER_ITERS: usize = 50;

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "pca" => Some(builtin_pca(args)),
        _ => None,
    }
}

/// `pca(X, k) -> Y [N, k]`. Orchestrator: validate,
/// center, covariance, extract top-k components, then
/// `Y = Xc @ V^T` row by row (V is `[k, D]` row-major so
/// its transpose is `[D, k]`).
fn builtin_pca(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate_pca_args(args)?;
    let xc = center_data(&xs, n, d);
    let cov = compute_cov(&xc, n, d);
    let v = extract_components(cov, d, k);
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

fn validate_pca_args(
    args: Vec<DenseArray>,
) -> Result<(usize, usize, usize, Vec<f64>), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: "pca".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let x = &args[0];
    if x.rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: "pca".into(),
            reason: format!("X must be rank-2 [N, D], got rank {}", x.rank()),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "pca".into(),
            reason: "k must be a scalar".into(),
        });
    }
    let k_f = args[1].data()[0];
    if !k_f.is_finite() || k_f <= 0.0 || k_f.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "pca".into(),
            reason: format!("k must be a positive integer, got {k_f}"),
        });
    }
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    let k = k_f as usize;
    if k > d {
        return Err(RuntimeError::InvalidArgument {
            func: "pca".into(),
            reason: format!("k = {k} must be <= D = {d}"),
        });
    }
    let xs = x.data().to_vec();
    if !xs.iter().all(|v| v.is_finite()) {
        return Err(RuntimeError::InvalidArgument {
            func: "pca".into(),
            reason: "X must contain only finite values (no NaN/Inf)".into(),
        });
    }
    Ok((n, d, k, xs))
}

/// Column-center `xs [N, D]`: subtract per-column mean
/// from every row. Returns a fresh flat buffer.
fn center_data(xs: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut means = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            means[j] += xs[i * d + j];
        }
    }
    let n_f = n as f64;
    for m in &mut means {
        *m /= n_f;
    }
    let mut xc = vec![0.0_f64; n * d];
    for i in 0..n {
        for j in 0..d {
            xc[i * d + j] = xs[i * d + j] - means[j];
        }
    }
    xc
}

/// `Cov = Xc^T Xc / N`, `[D, D]` row-major.
fn compute_cov(xc: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; d * d];
    let n_f = n as f64;
    for a in 0..d {
        for b in a..d {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += xc[i * d + a] * xc[i * d + b];
            }
            let v = s / n_f;
            cov[a * d + b] = v;
            cov[b * d + a] = v;
        }
    }
    cov
}

/// Extract top-k principal components from `cov [D, D]`
/// via power iteration with Gram-Schmidt orthogonalization
/// against prior components + deflation. The GS step is
/// what guarantees orthogonality when later components'
/// eigenvalues are numerical noise (deflation alone would
/// let v drift back toward earlier components). Returns
/// `V [k, D]` row-major: row `i` is the `i`-th component.
fn extract_components(mut cov: Vec<f64>, d: usize, k: usize) -> Vec<f64> {
    let mut components = vec![0.0_f64; k * d];
    let mut v = vec![0.0_f64; d];
    let mut next = vec![0.0_f64; d];
    for comp in 0..k {
        // Seed at a basis vector orthogonal to priors.
        // Try e_comp, e_{comp+1}, ... until one survives
        // Gram-Schmidt with nonzero norm.
        for start in 0..d {
            v.iter_mut().for_each(|x| *x = 0.0);
            v[(comp + start) % d] = 1.0;
            orthogonalize(&mut v, &components, comp, d);
            let n0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n0 > 1e-12 {
                for vi in v.iter_mut() {
                    *vi /= n0;
                }
                break;
            }
        }
        for _ in 0..POWER_ITERS {
            for a in 0..d {
                let row = &cov[a * d..a * d + d];
                next[a] = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            }
            orthogonalize(&mut next, &components, comp, d);
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-30 {
                for (vi, ni) in v.iter_mut().zip(next.iter()) {
                    *vi = ni / norm;
                }
            }
        }
        let mut lambda = 0.0_f64;
        for a in 0..d {
            let row = &cov[a * d..a * d + d];
            let cv: f64 = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            lambda += v[a] * cv;
        }
        components[comp * d..comp * d + d].copy_from_slice(&v);
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] -= lambda * v[a] * v[b];
            }
        }
    }
    components
}

/// Subtract projections of `w` onto the first `comp`
/// rows of `components` (each a unit vector in `[k, D]`
/// row-major layout). In-place.
fn orthogonalize(w: &mut [f64], components: &[f64], comp: usize, d: usize) {
    for prior in 0..comp {
        let pv = &components[prior * d..prior * d + d];
        let dot: f64 = w.iter().zip(pv.iter()).map(|(a, b)| a * b).sum();
        for (wi, pi) in w.iter_mut().zip(pv.iter()) {
            *wi -= dot * pi;
        }
    }
}
