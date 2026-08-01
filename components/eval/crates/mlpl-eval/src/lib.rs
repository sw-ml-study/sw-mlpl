//! Expression evaluator for MLPL.
//!
//! Saga 32 step 001: the leaf modules `model`, `metric_sink`,
//! and `inspect_groups` have been extracted into the new
//! `mlpl-eval-core` crate to start the structural paydown of
//! the 42-module mlpl-eval. This file re-exports them so
//! downstream `use mlpl_eval::ModelSpec` etc. is unchanged.

mod analysis_dispatch;
mod auto_tag;
// The env layer lives below in mlpl-eval-env (env-base-out step);
// re-exported as modules so every crate::env:: / crate::env_user_fns::
// path keeps resolving, and callers outside see mlpl_eval::Environment
// unchanged.
pub use mlpl_eval_env::env;
// The Model DSL cluster lives in mlpl-eval-models (cluster peel 1);
// the module re-export keeps every crate::model_dispatch:: caller
// (fncall dispatch, auto_tag) working unchanged.
pub(crate) use mlpl_eval_models::{
    model_dispatch, model_feasibility, model_freeze, model_inspect, model_io, model_lora,
    model_mutate,
};
/// The capability-trait prelude: importing this glob gives every
/// `Environment` method its trait (the eval decomposition converts
/// env_* inherent impls to per-capability trait crates). External
/// crates use `mlpl_eval::env_api::*`; internal modules use
/// `crate::env_api::*`.
pub mod env_api {
    pub use mlpl_eval_envc_bindings::{
        EnvRecords, EnvResults, EnvScope, EnvStringLists, EnvStrings, EnvVars, ScopeSnapshot,
    };
    pub use mlpl_eval_envc_exec::{
        EnvDevice, EnvDeviceNotices, EnvDeviceTensors, EnvInterrupt, EnvPeer, EnvTensorDevice,
    };
    pub use mlpl_eval_envc_ml::{
        EnvFrozen, EnvMlIters, EnvModels, EnvParams, EnvTags, EnvTokenizers,
    };
    pub use mlpl_eval_envc_obs::{
        EnvBuiltinRefs, EnvDataDir, EnvExpDir, EnvExpLog, EnvMetricEmit, EnvMetricSink,
    };
    pub use mlpl_eval_envc_userfns::{EnvUserFns, EnvUserFnsRender};
}
pub use env_api::*;
mod bpe;
mod device;
mod device_dispatch;
mod device_to;
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
mod fncall_arrays;
mod fncall_axes;
mod fncall_engram;
mod fncall_engram_args;
mod fncall_models;
mod fncall_trace;
mod grad;
mod grad_calls_basic;
mod grad_calls_engram;
mod grad_calls_shape;
mod grad_optim;
// The GPU optimizer-step seam types (GpuEnv, GpuAdamStep, AdamHp,
// layouts, the step registry) moved to mlpl-eval-state
// (env-types-out step); the cfg-gated re-exports below preserve the
// public GPU-build surface.
// The GPU LoRA/MLP COMPUTE moved to sibling crates -- CUDA to
// mlpl-cuda-eval (S3), MLX to mlpl-mlx-eval (S4). Only the device-agnostic
// architecture RECOGNIZERS stay here (they read ModelSpecs + eval the X/Y
// exprs, so they are interpreter-coupled); the seam (gpu_step) hands the
// recognized layout to whichever backend the binary registered.
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
mod inspect;
mod inspect_collections;
mod inspect_describe;
mod inspect_introspect;
mod inspect_list;
mod inspect_render;
mod interrupt;
mod llm_dispatch;
mod loader;
mod result_ops;
mod tag_propagate;
mod tag_render;
mod tokenizer;
mod type_errors;

pub use env::{Environment, PeerDispatcher, model_params};
pub use eval_program::{
    eval_program, eval_program_traced, eval_program_value, eval_program_value_traced,
    eval_source_value,
};
pub use experiment::{ExperimentRecord, ParamShape};
pub use grad::{OptimizerState, optim_state, optim_state_mut};
// The GPU optimizer-step seam (S1-S3): the public surface the sibling
// `mlpl-cuda-eval` / `mlpl-mlx-eval` compute crates build against -- the
// step trait, the narrow env accessor, the hyperparameters, the resolved
// layouts (the recognizers stay here, interpreter-coupled), plus the
// registration the binary drives. Only present on a GPU build.
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub use grad_optim_mlx_demo::DemoLayout;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub use grad_optim_mlx_mlp::LoraNames;
pub use inspect::{colon_ref_hint, inspect, is_colon_call_expr};
pub use interrupt::Interrupt;
pub use mlpl_eval_core::inspect_groups::documented_builtin_names;
pub use mlpl_eval_core::{ActKind, MetricSink, ModelSpec};
#[cfg(feature = "image-io")]
pub use mlpl_eval_image::decode_and_resize_u8;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub use mlpl_eval_models::tokens_to_onehot;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub use mlpl_eval_state::register_gpu_step;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub use mlpl_eval_state::{AdamHp, GpuAdamStep, GpuEnv};
pub use mlpl_eval_types::EvalError;
pub use mlpl_eval_types::{Value, value_kind};
pub use mlpl_runtime::runtime_builtin_names;
pub use tokenizer::TokenizerSpec;
