//! Built-in function registry and dispatch.
//!
//! Most array/shape/reduce/slice builtins live in
//! `mlpl-runtime-array`; the rest are dispatched through the
//! `try_external_dispatchers` chain below. This module only
//! holds the dispatch glue plus the `grid` / `concat` (2-arg)
//! arms that don't fit the per-crate `try_call` pattern.

use mlpl_array::DenseArray;

use mlpl_runtime_core::error::RuntimeError;

/// Names handled by the local match in `match_local_builtin`
/// below (i.e., names NOT covered by any per-crate `try_call`).
/// Currently just `grid`, which dispatches into mlpl-runtime-data.
pub(crate) const LOCAL_NAMES: &[&str] = &["grid"];

/// Try every external per-module `try_call` dispatcher in
/// registration order; returns `Some(result)` for the first
/// module that recognizes `name`.
fn try_external_dispatchers(
    name: &str,
    args: &[DenseArray],
) -> Option<Result<DenseArray, RuntimeError>> {
    mlpl_runtime_math::try_call(name, args.to_vec())
        .or_else(|| mlpl_runtime_array::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_conv::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_rnn::try_call(name, args.to_vec()))
        .or_else(|| crate::random_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_data::dataset_builtins::try_call(name, args.to_vec()))
        .or_else(|| crate::ml_builtins::try_call(name, args.to_vec()))
        .or_else(|| crate::ensemble_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_data::embedding_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_dim_reduction::tsne_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_dim_reduction::pca_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_umap::umap_graph::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_umap::umap_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_mds_rp::mds::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_mds_rp::random_projection::try_call(name, args.to_vec()))
}

/// Dispatch a built-in function call by name.
pub fn call_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if let Some(result) = try_external_dispatchers(name, &args) {
        return result;
    }
    match_local_builtin(name, args)
}

/// Match the local in-crate builtin names. Currently just `grid`
/// (which lives in mlpl-runtime-data but isn't behind a `try_call`).
fn match_local_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    match name {
        "grid" => mlpl_runtime_data::grid_builtin::builtin_grid(name, args),
        _ => Err(RuntimeError::UnknownFunction(name.into())),
    }
}
