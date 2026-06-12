//! The device-agnostic GPU optimizer-step seam.
//!
//! Only compiled when a GPU backend is configured (CUDA on
//! linux/x86_64, MLX on macos/aarch64). The interpreter-coupled
//! RECOGNITION (`demo_layout` / `mlp_layout` read models; `extract_xy`
//! calls `eval_expr`) stays in `grad_optim::eval_adam`; this seam gives
//! the GPU compute only the recognized layout + input tensors + a narrow
//! [`GpuEnv`] accessor, so the impls reference no interpreter / ModelSpec
//! and can move to sibling cuda/mlx crates. See
//! docs/build-and-workspace-plan.md.

use std::sync::Arc;

use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;

use crate::grad_optim_mlx_demo::DemoLayout;
use crate::grad_optim_mlx_mlp::LoraNames;

// The environment accessor lives in its own module; re-exported here so
// `crate::gpu_step::GpuEnv` keeps resolving for the backend compute.
pub use crate::gpu_env::GpuEnv;

/// Device-agnostic Adam hyperparameters; each backend converts to its
/// own (`mlpl_cuda_train` / `mlpl_mlx_train`).
#[derive(Clone, Copy, Debug)]
pub struct AdamHp {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub t: i32,
}

/// One GPU optimizer step for a RECOGNIZED architecture. Recognition is
/// done by the caller (`eval_adam`); the impl gets the resolved layout +
/// input tensors + the [`GpuEnv`] accessor, and returns the step loss.
pub trait GpuAdamStep: std::fmt::Debug + Send + Sync {
    /// Head-only LoRA fine-tune step.
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError>;
    /// Board-policy MLP step (`Chain[LinearLora, relu, LinearLora]`).
    fn run_mlp_step(
        &self,
        l1: &LoraNames,
        head: &LoraNames,
        x: &DenseArray,
        y: &DenseArray,
        hp: &AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError>;
}

/// The CUDA step moved to the sibling `mlpl-cuda-eval` crate (S3), so
/// this crate has no in-crate CUDA default. The binary registers
/// `mlpl_cuda_eval::gpu_step()` via `register_gpu_step` at startup, so
/// `installed_gpu_step` returns the registered step; this `None` is only
/// the fallback when nothing is registered (e.g. a CUDA build that never
/// calls the registration path).
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn default_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    None
}

/// The MLX step (mlx/macos/aarch64 build). Still in-crate until S4.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[must_use]
pub(crate) fn default_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    Some(Arc::new(crate::grad_optim_mlx::MlxGpuAdam))
}
