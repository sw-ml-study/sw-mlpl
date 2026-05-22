//! Saga 32 step 002: dataset / grid / embedding builtins
//! extracted from mlpl-runtime.
//!
//! - `dataset_builtins` (with `dataset_validate`): synthetic
//!   dataset generators (`circles`, `moons`, `swiss_roll`)
//!   plus the shared arg-validation helpers.
//! - `grid_builtin`: 2-D coordinate grid generator (`grid`).
//! - `embedding_builtins`: vector / cosine-similarity ops
//!   the embedding viz pipeline calls into (`norm_l2`,
//!   `top_k_cosine`).
//!
//! All three modules expose `pub const NAMES` + `pub fn
//! try_call` (or, for `grid_builtin`, the direct
//! `builtin_grid`). `mlpl-runtime`'s `builtins.rs` registry
//! dispatches to these.

pub mod dataset_builtins;
pub mod dataset_validate;
pub mod embedding_builtins;
pub mod grid_builtin;
