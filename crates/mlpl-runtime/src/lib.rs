//! Built-in function registry and dispatch for MLPL.

mod builtins;
mod dataset_builtins;
mod embedding_builtins;
mod ensemble_builtins;
mod error;
mod grid_builtin;
mod math_builtins;
mod ml_builtins;
mod prng;
mod random_builtins;
mod tsne_affinities;
mod tsne_builtin;
mod tsne_gradient;
mod tsne_validate;

pub use builtins::call_builtin;
pub use error::RuntimeError;
