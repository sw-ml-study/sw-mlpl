//! Model-mutating analyses extracted from
//! `crates/mlpl-eval/src/model_{clone,perturb}.rs`. Saga 33
//! step 012.
//!
//! Both ops follow the C+D pattern from step 008/011:
//! - Generic over `E: HasModels + HasVars + HasParams + ...`
//!   (the slices each fn actually needs).
//! - Generic `Err: From<MutateError>` so the sub-crate exports
//!   its own error vocabulary.
//! - Eval-loop dependencies (`eval_expr`, `scalar_f64`) are
//!   injected as caller-supplied closures, so this crate never
//!   imports the wider eval engine.

pub mod clone;
pub mod clone_attention;
pub mod clone_variants;
pub mod error;
pub mod perturb;
pub mod perturb_helpers;

pub use clone::{clone_model_inner, clone_spec};
pub use error::MutateError;
pub use perturb::perturb_params_inner;
