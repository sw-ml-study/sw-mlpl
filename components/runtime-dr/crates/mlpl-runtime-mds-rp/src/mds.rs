//! Saga 33 step 041: `mds(X, k, iters, seed)` Multidimensional
//! Scaling via stress-minimization SGD.
//!
//! Classical metric MDS finds low-D coordinates Y [N, k] whose
//! pairwise distances best preserve the input pairwise
//! distances. The SGD variant minimizes the stress function
//!   `S = sum_{i<j} (||Y_i - Y_j|| - d_ij)^2`
//! by sampling random pairs each iteration. Reuses the same
//! Xorshift64 PRNG + linear-decay learning rate schedule as
//! t-SNE / UMAP's optimization loops. Simpler than the
//! classical eigendecomposition approach because the input
//! pairwise distance matrix is the only thing we need to
//! precompute -- no double-centering, no eigenvectors.

use mlpl_array::{DenseArray, Shape};

use mlpl_runtime_core::error::RuntimeError;
use mlpl_runtime_core::prng::Xorshift64;

pub const NAMES: &[&str] = &["mds"];

const LR0: f64 = 0.5;
const INIT_SCALE: f64 = 1.0;
const EPS: f64 = 1e-6;

/// Parsed `mds` args: `(n, d, k, iters, seed, xs)`.
type ValidatedArgs = (usize, usize, usize, usize, f64, Vec<f64>);

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "mds" => Some(builtin_mds(args)),
        _ => None,
    }
}

/// `mds(X, k, iters, seed) -> [N, k]`. Orchestrator.
fn builtin_mds(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, iters, seed, xs) = validate(args)?;
    let dist = pairwise_dist(&xs, n, d);
    let mut y = init_y(n, k, seed);
    run_sgd(&mut y, &dist, n, k, iters, seed);
    Ok(DenseArray::new(Shape::new(vec![n, k]), y)?)
}

/// Precompute the `[N, N]` pairwise Euclidean distance
/// matrix (not squared -- the SGD gradient is in terms of
/// linear distance). Symmetric, zero diagonal.
fn pairwise_dist(xs: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s: f64 = (0..d)
                .map(|c| (xs[i * d + c] - xs[j * d + c]).powi(2))
                .sum();
            let v = s.sqrt();
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    out
}

fn init_y(n: usize, k: usize, seed: f64) -> Vec<f64> {
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed.max(1));
    (0..n * k)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) * INIT_SCALE)
        .collect()
}

/// Stress-minimization SGD. Each iter samples `N` random
/// pairs and nudges them toward their target distance. The
/// gradient of `(||Y_i - Y_j|| - d_ij)^2` w.r.t. `Y_i` is
/// `2 * (||Y_i - Y_j|| - d_ij) * (Y_i - Y_j) / ||Y_i - Y_j||`.
fn run_sgd(y: &mut [f64], dist: &[f64], n: usize, k: usize, iters: usize, seed: f64) {
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed.wrapping_add(0x_C0DE_BABE));
    let total = iters.max(1) as f64;
    for t in 0..iters {
        let lr = LR0 * (1.0 - (t as f64) / total);
        for _ in 0..n.saturating_mul(n) {
            let i = (rng.next_f64() * n as f64) as usize % n.max(1);
            let j = (rng.next_f64() * n as f64) as usize % n.max(1);
            if i == j {
                continue;
            }
            sgd_pair_step(y, dist[i * n + j], (i, j), k, lr);
        }
    }
}

/// One stress-majorization SGD update for the pair `(i, j)`: pull
/// or push the two embedded points along their difference vector
/// toward the target distance.
fn sgd_pair_step(y: &mut [f64], target: f64, (i, j): (usize, usize), k: usize, lr: f64) {
    let mut cur_sq = 0.0_f64;
    for c in 0..k {
        let d_c = y[i * k + c] - y[j * k + c];
        cur_sq += d_c * d_c;
    }
    let cur = cur_sq.sqrt().max(EPS);
    let g = lr * (cur - target) / cur;
    for c in 0..k {
        let delta = g * (y[i * k + c] - y[j * k + c]);
        y[i * k + c] -= delta;
        y[j * k + c] += delta;
    }
}

fn validate(args: Vec<DenseArray>) -> Result<ValidatedArgs, RuntimeError> {
    if args.len() != 4 {
        return Err(RuntimeError::ArityMismatch {
            func: "mds".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let bad = |r: String| RuntimeError::InvalidArgument {
        func: "mds".into(),
        reason: r,
    };
    let x = &args[0];
    if x.rank() != 2 {
        return Err(bad(format!(
            "X must be rank-2 [N, D], got rank {}",
            x.rank()
        )));
    }
    if !args[1..].iter().all(|a| a.rank() == 0) {
        return Err(bad("k / iters / seed must be scalars".into()));
    }
    let (kf, itf, seed) = (args[1].data()[0], args[2].data()[0], args[3].data()[0]);
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    if !(kf.is_finite() && kf > 0.0 && kf.fract() == 0.0) {
        return Err(bad(format!("k must be a positive integer, got {kf}")));
    }
    if !(itf.is_finite() && itf >= 0.0 && itf.fract() == 0.0) {
        return Err(bad(format!(
            "iters must be a non-negative integer, got {itf}"
        )));
    }
    Ok((n, d, kf as usize, itf as usize, seed, x.data().to_vec()))
}
