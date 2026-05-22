//! Saga 32 step 002: dimension-reduction builtins extracted
//! from mlpl-runtime.
//!
//! Self-contained subtree: t-SNE (validate, affinities,
//! gradient, builtin orchestrator) + PCA. Both expose a
//! `try_call(name, args)` dispatcher that returns
//! `Some(Result<DenseArray, RuntimeError>)` when the name is
//! one of their builtins and `None` otherwise.
//!
//! `mlpl-runtime`'s `builtins.rs` registry delegates to these
//! dispatchers. The crate depends on `mlpl-runtime-core` (for
//! `RuntimeError` + `Xorshift64`) and `mlpl-array`; nothing
//! depends back up the dep graph, so no cycles.

pub mod pca_builtin;
pub mod tsne_affinities;
pub mod tsne_builtin;
pub mod tsne_gradient;
pub mod tsne_validate;
