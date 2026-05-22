//! Built-in function registry and dispatch for MLPL.
//!
//! Saga 32 step 002: `error` + `prng` extracted into
//! `mlpl-runtime-core`; the t-SNE + PCA family extracted into
//! `mlpl-runtime-dim-reduction`. This crate re-exports `RuntimeError`
//! so downstream `use mlpl_runtime::RuntimeError` keeps working.

mod builtins;
mod ensemble_builtins;
mod llm_builtins;
mod math_builtins;
mod ml_builtins;
mod random_builtins;

pub use builtins::call_builtin;
pub use llm_builtins::call_ollama;
pub use mlpl_runtime_core::RuntimeError;

/// Iterate every builtin name dispatched by [`call_builtin`].
/// Aggregates the per-module `NAMES` constants plus
/// [`builtins::LOCAL_NAMES`]. Used by the help-completeness
/// test in `mlpl-eval/tests/help_completeness_tests.rs` to
/// catch undocumented runtime builtins.
pub fn runtime_builtin_names() -> impl Iterator<Item = &'static str> {
    builtins::LOCAL_NAMES
        .iter()
        .chain(math_builtins::NAMES)
        .chain(random_builtins::NAMES)
        .chain(mlpl_runtime_data::dataset_builtins::NAMES)
        .chain(ml_builtins::NAMES)
        .chain(ensemble_builtins::NAMES)
        .chain(mlpl_runtime_data::embedding_builtins::NAMES)
        .chain(mlpl_runtime_dim_reduction::tsne_builtin::NAMES)
        .chain(mlpl_runtime_dim_reduction::pca_builtin::NAMES)
        .copied()
}
