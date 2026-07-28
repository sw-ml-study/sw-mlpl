//! Leaf session-state types shared BELOW the evaluator hub
//! (eval-decomposition saga, env-types-out step; design in
//! docs/eval-env-design.md). These are the types `Environment`'s
//! fields reference; moving them here lets the env layer -- and
//! later every evaluator cluster -- live in crates below
//! `mlpl-eval` without cycles.
//!
//! - `run_state`: interrupt token, optimizer buffers, experiment
//!   records.
//! - `gpu_env` / `gpu_step`: the GPU optimizer-step seam (narrow
//!   env accessor trait, step trait + hyperparameters + recognized
//!   layouts, process-global step registry).

pub mod gpu_env;
pub mod gpu_step;
pub mod run_state;

pub use gpu_env::GpuEnv;
pub use gpu_step::{
    AdamHp, DemoLayout, GpuAdamStep, LoraNames, installed_gpu_step, register_gpu_step,
};
pub use run_state::{ExperimentRecord, Interrupt, OptimizerState, ParamShape};
