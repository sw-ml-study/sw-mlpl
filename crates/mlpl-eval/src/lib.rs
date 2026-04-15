//! Expression evaluator for MLPL.

mod env;
mod error;
mod eval;
mod eval_for;
mod eval_ops;
mod grad;
mod inspect;
mod loader;
mod model;
mod model_dispatch;
mod model_tape;
mod value;

pub use env::{Environment, model_params};
pub use error::EvalError;
pub use eval::{eval_program, eval_program_traced, eval_program_value};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
pub use inspect::inspect;
pub use model::ModelSpec;
pub use value::Value;
