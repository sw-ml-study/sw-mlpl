//! The device-agnostic GPU optimizer-step seam.
//!
//! Only compiled when a GPU backend is configured (CUDA on
//! linux/x86_64, MLX on macos/aarch64) -- on a CPU-only build this
//! module does not exist, so there is no unused GPU machinery. The
//! interpreter (`grad_optim::eval_adam`) calls `GpuAdamStep` without
//! referencing any backend crate directly; the impls live in the
//! feature-gated `grad_optim_cuda` / `grad_optim_mlx` modules and can
//! later move to sibling cuda/mlx crates. See
//! docs/build-and-workspace-plan.md + future-saga-gpu-training.md.

use std::sync::Arc;

use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

use crate::env::Environment;

/// Device-agnostic Adam hyperparameters; each backend converts to its
/// own (`mlpl_cuda_train` / `mlpl_mlx_train`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdamHp {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub t: i32,
}

/// A GPU-resident optimizer step for a recognized model architecture,
/// implemented by the CUDA / MLX backends. `None` means "not this
/// architecture / device" -> the CPU autograd tape handles it.
pub(crate) trait GpuAdamStep: std::fmt::Debug + Send + Sync {
    /// Head-only LoRA fine-tune step.
    fn try_lora_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>>;
    /// Board-policy MLP (`Chain[LinearLora, relu, LinearLora]`) step.
    fn try_mlp_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>>;
}

/// The CUDA step (cuda/linux/x86_64 build).
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn default_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    Some(Arc::new(crate::grad_optim_cuda::CudaGpuAdam))
}

/// The MLX step (mlx/macos/aarch64 build).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[must_use]
pub(crate) fn default_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    Some(Arc::new(crate::grad_optim_mlx::MlxGpuAdam))
}
