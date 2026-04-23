//! Saga 16 step 001: `pairwise_sqdist` + `knn` array-
//! utility builtins.
//!
//! Two sibling distance-based ops that close the
//! "inspect-an-embedding-table" surface:
//! - `pairwise_sqdist(X [N, D]) -> D2 [N, N]` squared
//!   Euclidean distances.
//! - `knn(X [N, D], k) -> idx [N, k]` indices of the k
//!   nearest non-self neighbors per row, sorted by
//!   ascending distance, ties broken by lower original
//!   index.
//!
//! See `contracts/eval-contract/pairwise-sqdist.md` and
//! `contracts/eval-contract/knn.md`.

use std::cmp::Ordering;

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "pairwise_sqdist" => Some(builtin_pairwise_sqdist(args)),
        "knn" => Some(builtin_knn(args)),
        _ => None,
    }
}

/// `pairwise_sqdist(X) -> [N, N]` squared Euclidean
/// distances. `D[i, j] = sum_k (X[i, k] - X[j, k])^2`.
fn builtin_pairwise_sqdist(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: "pairwise_sqdist".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let x = &args[0];
    if x.rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: "pairwise_sqdist".into(),
            reason: format!("X must be rank-2 [N, D], got rank {}", x.rank()),
        });
    }
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    let xs = x.data();
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        // Only compute the upper triangle; mirror for
        // lower. Diagonal stays 0.
        for j in (i + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..d {
                let diff = xs[i * d + k] - xs[j * d + k];
                s += diff * diff;
            }
            out[i * n + j] = s;
            out[j * n + i] = s;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n, n]), out)?)
}

/// `knn(X, k) -> [N, k]` integer-valued indices of the k
/// nearest non-self neighbors per row.
fn builtin_knn(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate_knn_args(args)?;
    let mut scratch: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
    let mut out = vec![0.0_f64; n * k];
    for i in 0..n {
        scratch.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut s = 0.0_f64;
            for kk in 0..d {
                let diff = xs[i * d + kk] - xs[j * d + kk];
                s += diff * diff;
            }
            scratch.push((s, j));
        }
        // Stable sort: distance ascending, then index
        // ascending. NaN (partial_cmp == None) is treated
        // as equal so the stable sort preserves
        // insertion (ascending-index) order.
        scratch.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
            Some(Ordering::Equal) | None => a.1.cmp(&b.1),
            Some(o) => o,
        });
        for pos in 0..k {
            out[i * k + pos] = scratch[pos].1 as f64;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n, k]), out)?)
}

/// Validate `knn`'s arguments + convert them into the
/// fixed shape the inner loop needs. Returns
/// `(N, D, k, xs)` where `xs` is the flat row-major data
/// of `X`. Kept as a struct-return helper so
/// `builtin_knn` stays short and every precondition
/// error lives in one place.
fn validate_knn_args(
    args: Vec<DenseArray>,
) -> Result<(usize, usize, usize, Vec<f64>), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: "knn".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let x = &args[0];
    if x.rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: "knn".into(),
            reason: format!("X must be rank-2 [N, D], got rank {}", x.rank()),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "knn".into(),
            reason: "k must be a scalar".into(),
        });
    }
    let k_f = args[1].data()[0];
    if !k_f.is_finite() || k_f <= 0.0 || k_f.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "knn".into(),
            reason: format!("k must be a positive integer, got {k_f}"),
        });
    }
    let k = k_f as usize;
    let dims = x.shape().dims();
    let (n, d) = (dims[0], dims[1]);
    if k >= n {
        return Err(RuntimeError::InvalidArgument {
            func: "knn".into(),
            reason: format!("k = {k} must be < N = {n} (self is excluded, leaving N-1 candidates)"),
        });
    }
    Ok((n, d, k, x.data().to_vec()))
}
