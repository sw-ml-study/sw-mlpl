//! UMAP data-structure layer. Saga 33 step 033 (Phase 2a of
//! the dim-reduction milestone): the `knn_graph(X, k)` builtin.
//!
//! Returns the k-nearest-neighbor edge list of `X [N, D]` as
//! `[N*k, 3]`, with rows `(i, j, dist)` where `dist` is the
//! Euclidean distance from point `i` to its `p`-th nearest
//! non-self neighbor (sorted ascending). The mlpl-runtime-data
//! crate already ships `knn(X, k) -> [N, k]` returning just
//! indices; UMAP's downstream optimization needs distances
//! too, plus the explicit `i` column to make it easy to scan
//! a flat edge list without re-deriving the source per row.
//!
//! Brute-force scan is intentional -- the milestone targets
//! N <= 5000, D <= 1024, k <= 30, which is well within budget
//! for a tree-walked Rust scan. Approximate-NN structures
//! (HNSW, IVF) are deferred per the milestone non-goals.
//!
//! `fuzzy_simplicial_set` (the local-sigma + symmetrization
//! step) is deferred to step 034 where it lands alongside the
//! layout SGD; the two are tightly coupled in UMAP's
//! optimization loop and benefit from sharing
//! validation / data-structure code.

use std::cmp::Ordering;

use mlpl_array::{DenseArray, Shape};

use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &["knn_graph"];

/// Bundled inputs threaded into [`write_nearest_k`] so the
/// per-source-row helper keeps a 4-arg signature (vs. 7 raw
/// scalars), staying inside the function-LOC budget.
struct Ctx<'a> {
    n: usize,
    d: usize,
    k: usize,
    xs: &'a [f64],
}

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "knn_graph" => Some(builtin_knn_graph(args)),
        _ => None,
    }
}

/// `knn_graph(X, k) -> [N*k, 3]`. Edge list of the k nearest
/// non-self neighbors per row of `X`, sorted by ascending
/// distance with index ties broken by lower neighbor id.
fn builtin_knn_graph(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (n, d, k, xs) = validate(args)?;
    let ctx = Ctx { n, d, k, xs: &xs };
    let mut out = vec![0.0_f64; n * k * 3];
    let mut scratch: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
    for i in 0..n {
        write_nearest_k(i, &ctx, &mut scratch, &mut out);
    }
    Ok(DenseArray::new(Shape::new(vec![n * k, 3]), out)?)
}

/// Fill `out`'s k consecutive rows for source `i` with the k
/// nearest non-self neighbors by Euclidean distance.
fn write_nearest_k(i: usize, c: &Ctx<'_>, scratch: &mut Vec<(f64, usize)>, out: &mut [f64]) {
    scratch.clear();
    scratch.extend((0..c.n).filter(|&j| j != i).map(|j| {
        let sq: f64 = (0..c.d)
            .map(|p| (c.xs[i * c.d + p] - c.xs[j * c.d + p]).powi(2))
            .sum();
        (sq, j)
    }));
    scratch.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
        Some(Ordering::Equal) | None => a.1.cmp(&b.1),
        Some(o) => o,
    });
    for (pos, &(sq, j)) in scratch.iter().take(c.k).enumerate() {
        let row = (i * c.k + pos) * 3;
        out[row..row + 3].copy_from_slice(&[i as f64, j as f64, sq.sqrt()]);
    }
}

fn validate(args: Vec<DenseArray>) -> Result<(usize, usize, usize, Vec<f64>), RuntimeError> {
    let bad = |reason: String| RuntimeError::InvalidArgument {
        func: "knn_graph".into(),
        reason,
    };
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: "knn_graph".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let (x, kx) = (&args[0], &args[1]);
    let k_f = kx.data().first().copied().unwrap_or(f64::NAN);
    let k_ok = kx.rank() == 0 && k_f.is_finite() && k_f > 0.0 && k_f.fract() == 0.0;
    if !k_ok {
        return Err(bad(format!(
            "k must be a positive integer scalar, got {k_f}"
        )));
    }
    if x.rank() != 2 {
        return Err(bad(format!(
            "X must be rank-2 [N, D], got rank {}",
            x.rank()
        )));
    }
    let (k, dims) = (k_f as usize, x.shape().dims());
    let (n, d) = (dims[0], dims[1]);
    if k >= n {
        return Err(bad(format!("k = {k} must be < N = {n} (self excluded)")));
    }
    if !x.data().iter().all(|v| v.is_finite()) {
        return Err(bad("X must contain only finite values (no NaN/Inf)".into()));
    }
    Ok((n, d, k, x.data().to_vec()))
}
