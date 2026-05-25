//! UMAP builtins for MLPL. Saga 33 step 033 carved this out of
//! `mlpl-runtime-dim-reduction` so the dim-reduction crate stays
//! under the Crate-Module-Count budget while UMAP grows
//! (fuzzy_simplicial_set + layout SGD arrive in step 034+).
//!
//! Self-contained subtree: each module exposes a `NAMES` constant
//! and a `try_call(name, args)` dispatcher used by
//! `mlpl-runtime`'s registry to delegate. Depends only on
//! `mlpl-array` and `mlpl-runtime-core`; nothing depends back
//! upward, so the dep graph stays a DAG.

pub mod umap_graph;
