//! Shared types for mlpl-eval: Value, EvalError, TokenizerSpec.
//! Extracted in saga 73 so god-crate decomposition (Phase 1
//! extractions) can depend on these without cycling through
//! mlpl-eval.

mod error;
mod error_fmt;
mod error_from_models;
mod error_from_tools;
mod error_kind;
mod value;

pub use error::EvalError;
pub use error_kind::error_kind;
pub use mlpl_eval_core::TokenizerSpec;
pub use value::{Value, value_kind};
