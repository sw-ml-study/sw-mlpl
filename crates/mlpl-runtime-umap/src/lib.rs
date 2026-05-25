//! UMAP builtins for MLPL. Saga 33 step 033 carved this out of
//! `mlpl-runtime-dim-reduction` so the dim-reduction crate stays
//! under the Crate-Module-Count budget while UMAP grows.
//!
//! Modules:
//! - `umap_graph` -- Phase 2a: `knn_graph(X, k)` k-NN edge list.
//! - `umap_simplicial` -- Phase 2b helper: fuzzy simplicial set
//!   from local sigma calibration + symmetric fuzzy union.
//! - `umap_layout` -- Phase 2b helper: SGD-based 2-D embedding
//!   optimization with negative sampling.
//! - `umap_builtin` -- Phase 2b orchestrator: the public
//!   `umap(X, n_neighbors, min_dist, iters, seed)` builtin.
//!
//! Each public module exposes a `NAMES` constant and a
//! `try_call(name, args)` dispatcher used by `mlpl-runtime`'s
//! registry to delegate. Depends only on `mlpl-array` and
//! `mlpl-runtime-core`; nothing depends back upward, so the dep
//! graph stays a DAG.

pub mod umap_builtin;
pub mod umap_graph;
mod umap_layout;
mod umap_simplicial;
