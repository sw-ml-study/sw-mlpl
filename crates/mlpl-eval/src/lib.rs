//! Expression evaluator for MLPL.

mod auto_tag;
mod bpe;
mod device;
mod env;
mod error;
mod eval;
mod eval_for;
mod eval_intercepts;
mod eval_ops;
mod eval_reduce;
mod experiment;
#[cfg(feature = "image-io")]
mod fetch_dataset;
mod grad;
#[cfg(feature = "image-io")]
mod image_io;
mod inspect;

#[cfg(feature = "image-io")]
pub use image_io::decode_and_resize_u8;
mod inspect_groups;
mod interrupt;
mod llm_dispatch;
mod loader;
mod metric_sink;
mod model;
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
pub use eval::{eval_program, eval_program_traced, eval_program_value};
pub use experiment::{ExperimentRecord, ParamShape};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
pub use inspect::inspect;
pub use inspect_groups::documented_builtin_names;
pub use interrupt::Interrupt;
pub use metric_sink::MetricSink;
pub use model::ModelSpec;
pub use tokenizer::TokenizerSpec;
pub use value::{Value, value_kind};
