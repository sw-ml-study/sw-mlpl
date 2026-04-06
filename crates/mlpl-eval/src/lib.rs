//! Expression evaluator for MLPL.

mod env;
mod error;
mod eval;

pub use env::Environment;
pub use error::EvalError;
pub use eval::{eval_program, eval_program_traced};
