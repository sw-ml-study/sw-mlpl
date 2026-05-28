//! Saga 33 step 041: `random_projection(X, k, seed)` -- the
//! Johnson-Lindenstrauss random projection.
//!
//! Builds an `[D, k]` Gaussian random matrix `R` (each entry
//! `~ N(0, 1)`), scales by `1/sqrt(k)`, then computes `X @ R`.
//! For modest `k = O(log N / eps^2)` the JL lemma guarantees
//! all pairwise distances are preserved within a `1 +- eps`
//! factor. The payoff: a principled-but-trivial dim reduction
//! that ships in a few lines, useful as a sanity baseline
//! against learned methods like PCA / t-SNE / UMAP.

use mlpl_array::{DenseArray, Shape};

use mlpl_runtime_core::error::RuntimeError;
use mlpl_runtime_core::prng::Xorshift64;

pub const NAMES: &[&str] = &["random_projection"];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "random_projection" => Some(builtin_random_projection(args)),
        _ => None,
    }
}

/// `random_projection(X, k, seed) -> [N, k]`. Builds a seeded
/// Gaussian random matrix and multiplies. Deterministic given
/// the same seed.
fn builtin_random_projection(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, seed, xs) = validate(args)?;
    let scale = 1.0 / (k as f64).sqrt();
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed.max(1));
    let r: Vec<f64> = (0..d * k).map(|_| rng.next_normal() * scale).collect();
    let mut out = vec![0.0_f64; n * k];
    for i in 0..n {
        for c in 0..k {
            let mut s = 0.0_f64;
            for p in 0..d {
                s += xs[i * d + p] * r[p * k + c];
            }
            out[i * k + c] = s;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n, k]), out)?)
}

fn validate(args: Vec<DenseArray>) -> Result<(usize, usize, usize, f64, Vec<f64>), RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: "random_projection".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let bad = |r: String| RuntimeError::InvalidArgument {
        func: "random_projection".into(),
        reason: r,
    };
    let x = &args[0];
    if x.rank() != 2 {
        return Err(bad(format!(
            "X must be rank-2 [N, D], got rank {}",
            x.rank()
        )));
    }
    if args[1].rank() != 0 || args[2].rank() != 0 {
        return Err(bad("k and seed must be scalars".into()));
    }
    let (kf, seed) = (args[1].data()[0], args[2].data()[0]);
    if !(kf.is_finite() && kf > 0.0 && kf.fract() == 0.0) {
        return Err(bad(format!("k must be a positive integer, got {kf}")));
    }
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    Ok((n, d, kf as usize, seed, x.data().to_vec()))
}
