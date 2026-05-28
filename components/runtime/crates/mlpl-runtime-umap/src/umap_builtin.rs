//! Saga 33 step 034: `umap(X, n_neighbors, min_dist, iters,
//! seed) -> Y [N, 2]` public entry + orchestrator.
//!
//! Composes Phase 2a (`umap_graph::knn_graph` -> `[N*k, 3]`
//! edge list) with Phase 2b (fuzzy simplicial set + layout
//! SGD) into a single builtin. The orchestrator only owns
//! arg validation + glue; every load-bearing computation
//! lives in a sibling module.
//!
//! Matches `tsne(X, perplexity, iters, seed)`'s API shape so
//! the comparison demos in Phase 3 can swap the two with a
//! one-line edit.

use mlpl_array::{DenseArray, Shape};

use mlpl_runtime_core::error::RuntimeError;

use crate::umap_graph;
use crate::umap_layout::{init_layout, run_layout_sgd};
use crate::umap_simplicial::fuzzy_simplicial_set;

pub const NAMES: &[&str] = &["umap"];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "umap" => Some(builtin_umap(args)),
        _ => None,
    }
}

/// `umap(X, n_neighbors, min_dist, iters, seed) -> [N, 2]`.
fn builtin_umap(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let (x, n_neighbors, min_dist, iters, seed) = validate(args)?;
    let n = x.shape().dims()[0];
    let k_arr = DenseArray::from_scalar(n_neighbors as f64);
    let knn_da = umap_graph::try_call("knn_graph", vec![x, k_arr])
        .ok_or_else(|| RuntimeError::UnknownFunction("knn_graph".into()))??;
    let edges = fuzzy_simplicial_set(knn_da.data(), n, n_neighbors);
    let mut y = init_layout(n, seed);
    let _ = run_layout_sgd(&mut y, &edges, iters, seed, min_dist);
    Ok(DenseArray::new(Shape::new(vec![n, 2]), y)?)
}

fn validate(args: Vec<DenseArray>) -> Result<(DenseArray, usize, f64, usize, f64), RuntimeError> {
    if args.len() != 5 {
        return Err(RuntimeError::ArityMismatch {
            func: "umap".into(),
            expected: 5,
            got: args.len(),
        });
    }
    let bad = |r: String| RuntimeError::InvalidArgument {
        func: "umap".into(),
        reason: r,
    };
    let x = args[0].clone();
    if x.rank() != 2 || !args[1..].iter().all(|a| a.rank() == 0) {
        return Err(bad(
            "X must be rank-2 [N, D]; n_neighbors / min_dist / iters / seed must all be scalars"
                .into(),
        ));
    }
    let n = x.shape().dims()[0];
    let (kf, md, itf, seed) = (
        args[1].data()[0],
        args[2].data()[0],
        args[3].data()[0],
        args[4].data()[0],
    );
    if !(kf.is_finite() && kf > 0.0 && kf.fract() == 0.0 && (kf as usize) < n) {
        return Err(bad(format!(
            "n_neighbors must be positive int < N = {n}, got {kf}"
        )));
    }
    if !(itf.is_finite() && itf >= 0.0 && itf.fract() == 0.0) {
        return Err(bad(format!("iters must be non-negative int, got {itf}")));
    }
    Ok((x, kf as usize, md.max(0.0), itf as usize, seed))
}
