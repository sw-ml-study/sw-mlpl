//! Environment ML-STATE capabilities as traits (eval decomposition
//! capability peel; design in docs/eval-env-design.md): models,
//! trainable/frozen params, tokenizers, value tags, and their
//! whole-map iterators. Implemented for `mlpl_eval_env::Environment`
//! in each trait's own module (orphan rule); the hub re-exports them
//! through its `env_api` prelude.

pub mod env_frozen;
pub mod env_ml_iters;
pub mod env_models;
pub mod env_params;
pub mod env_tags;
pub mod env_tokenizers;

pub use env_frozen::EnvFrozen;
pub use env_ml_iters::EnvMlIters;
pub use env_models::EnvModels;
pub use env_params::EnvParams;
pub use env_tags::EnvTags;
pub use env_tokenizers::EnvTokenizers;
