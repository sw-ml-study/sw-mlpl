//! Built-in function registry and dispatch for MLPL.

mod builtins;
mod dataset_builtins;
mod dataset_validate;
mod embedding_builtins;
mod ensemble_builtins;
mod error;
mod grid_builtin;
mod llm_builtins;
mod math_builtins;
mod ml_builtins;
mod pca_builtin;
mod prng;
mod random_builtins;
mod tsne_affinities;
mod tsne_builtin;
mod tsne_gradient;
mod tsne_validate;

pub use builtins::call_builtin;
pub use error::RuntimeError;
pub use llm_builtins::call_ollama;

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
        .chain(dataset_builtins::NAMES)
        .chain(ml_builtins::NAMES)
        .chain(ensemble_builtins::NAMES)
        .chain(embedding_builtins::NAMES)
        .chain(tsne_builtin::NAMES)
        .chain(pca_builtin::NAMES)
        .copied()
}
