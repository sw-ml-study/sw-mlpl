//! Expression evaluator for MLPL.
//!
//! Saga 32 step 001: the leaf modules `model`, `metric_sink`,
//! and `inspect_groups` have been extracted into the new
//! `mlpl-eval-core` crate to start the structural paydown of
//! the 42-module mlpl-eval. This file re-exports them so
//! downstream `use mlpl_eval::ModelSpec` etc. is unchanged.

mod auto_tag;
mod bpe;
mod device;
mod env;
mod error;
mod eval;
mod eval_for;
mod eval_intercepts;
mod eval_loop;
mod eval_ops;
mod eval_program;
mod eval_reduce;
mod eval_script;
mod experiment;
mod experiment_compare;
#[cfg(feature = "image-io")]
mod fetch_dataset;
mod grad;
mod grad_optim;
#[cfg(feature = "image-io")]
mod image_io;
mod inspect;

#[cfg(feature = "image-io")]
pub use image_io::decode_and_resize_u8;
mod interrupt;
mod llm_dispatch;
mod loader;
mod model_clone;
mod model_dispatch;
mod model_embed_table;
mod model_estimate;
mod model_feasibility;
mod model_freeze;
mod model_lora;
mod model_perturb;
mod model_tape;
mod pets_tiny;
mod result_ops;
mod tag_propagate;
mod tag_render;
mod tokenizer;
mod type_errors;
mod value;

pub use env::{Environment, PeerDispatcher, model_params};
pub use error::EvalError;
pub use eval_program::{
    eval_program, eval_program_traced, eval_program_value, eval_program_value_traced,
};
pub use experiment::{ExperimentRecord, ParamShape};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
pub use inspect::inspect;
pub use interrupt::Interrupt;
pub use mlpl_eval_core::inspect_groups::documented_builtin_names;
pub use mlpl_eval_core::{MetricSink, ModelSpec};
pub use tokenizer::TokenizerSpec;
pub use value::{Value, value_kind};
