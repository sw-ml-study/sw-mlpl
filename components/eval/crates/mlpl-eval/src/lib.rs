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
mod device_dispatch;
mod device_to;
mod env;
mod env_builtin_refs;
mod env_device;
mod env_device_tensors;
mod env_dirs;
mod env_exp_log;
mod env_frozen;
mod env_interrupt;
mod env_metric_sink;
mod env_models;
mod env_params;
mod env_peer;
mod env_records;
mod env_scope;
mod env_string_lists;
mod env_strings;
mod env_tags;
mod env_tensor_device;
mod env_tokenizers;
mod env_trait_impls_devices;
mod env_trait_impls_dispatch;
mod env_trait_impls_models;
mod env_trait_impls_params;
mod env_trait_impls_strings;
mod env_trait_impls_vars;
mod env_user_fns;
mod env_vars;
mod eval;
mod eval_blocks;
mod eval_fncalls;
mod eval_for;
mod eval_intercepts;
mod eval_loop;
mod eval_ops;
mod eval_program;
mod eval_reduce;
mod eval_script;
mod eval_user_fn;
mod experiment;
mod experiment_compare;
#[cfg(feature = "image-io")]
mod fetch_dataset;
#[cfg(feature = "image-io")]
mod fetch_io;
mod fncall_arrays;
mod fncall_axes;
mod fncall_models;
mod fncall_trace;
mod grad;
mod grad_calls_basic;
mod grad_calls_shape;
mod grad_optim;
// Device-agnostic GPU optimizer-step seam (GpuAdamStep trait + AdamHp).
// Only exists when a GPU backend is configured, so a CPU-only build has
// no unused GPU machinery (no clippy overrides needed).
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
mod gpu_step;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod grad_optim_mlx;
// Device-agnostic demo-architecture recognition (ModelSpec + loss-expr
// parsing); shared by the MLX and CUDA LoRA fast paths.
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod grad_optim_cuda;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod grad_optim_cuda_mlp;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod grad_optim_cuda_step;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
mod grad_optim_mlx_demo;
// The board-policy MLP recognizer is device-agnostic (ModelSpec + names);
// shared by the MLX and CUDA MLP fast paths.
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
mod grad_optim_mlx_mlp;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod grad_optim_mlx_mlp_step;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod grad_optim_mlx_step;
mod inspect;
mod inspect_collections;
mod inspect_describe;
mod inspect_introspect;
mod inspect_render;
mod interrupt;
mod llm_dispatch;
mod loader;
mod model_apply;
mod model_apply_attention;
mod model_apply_compose;
mod model_apply_embed;
mod model_apply_lora;
mod model_apply_simple;
mod model_attn_weights;
mod model_dispatch;
mod model_dispatch_scalar;
mod model_eval_apply;
mod model_eval_attention;
mod model_eval_compose;
mod model_eval_layers;
mod model_feasibility;
mod model_freeze;
mod model_inspect;
mod model_io;
mod model_lora;
mod model_mutate;
mod result_ops;
mod tag_propagate;
mod tag_render;
mod tokenizer;
mod type_errors;

pub use env::{Environment, PeerDispatcher, model_params};
pub use eval_program::{
    eval_program, eval_program_traced, eval_program_value, eval_program_value_traced,
};
pub use experiment::{ExperimentRecord, ParamShape};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
pub use inspect::inspect;
pub use interrupt::Interrupt;
pub use mlpl_eval_core::inspect_groups::documented_builtin_names;
pub use mlpl_eval_core::{ActKind, MetricSink, ModelSpec};
#[cfg(feature = "image-io")]
pub use mlpl_eval_image::decode_and_resize_u8;
pub use mlpl_eval_types::EvalError;
pub use mlpl_eval_types::{Value, value_kind};
pub use mlpl_runtime::runtime_builtin_names;
pub use tokenizer::TokenizerSpec;
