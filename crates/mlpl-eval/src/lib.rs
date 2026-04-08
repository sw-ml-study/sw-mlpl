//! Expression evaluator for MLPL.

mod env;
mod error;
mod eval;
mod eval_ops;
mod grad;
mod value;

pub use env::Environment;
pub use error::EvalError;
pub use eval::{eval_program, eval_program_traced, eval_program_value};
pub use value::Value;
