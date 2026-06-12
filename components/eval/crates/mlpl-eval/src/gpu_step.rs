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

// Both GPU compute impls now live in sibling crates -- CUDA in
// mlpl-cuda-eval (S3), MLX in mlpl-mlx-eval (S4) -- so this crate
// constructs no `GpuAdamStep`. The binary registers the right one at
// startup via `register_gpu_step`; there is no in-crate default, so the
// GPU fast path requires registration (an unregistered GPU build falls
// back to the CPU tape with a one-time notice, see
// `grad_optim::eval_adam`).
