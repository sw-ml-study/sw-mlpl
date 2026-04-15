//! Expression evaluator for MLPL.

mod bpe;
mod env;
mod error;
mod eval;
mod eval_for;
mod eval_ops;
mod experiment;
mod grad;
mod inspect;
mod loader;
mod model;
mod model_dispatch;
mod model_tape;
mod tokenizer;
mod value;

pub use env::{Environment, model_params};
pub use error::EvalError;
pub use eval::{eval_program, eval_program_traced, eval_program_value};
pub use experiment::{ExperimentRecord, ParamShape};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
pub use inspect::inspect;
pub use model::ModelSpec;
pub use tokenizer::TokenizerSpec;
pub use value::Value;
