//! The device-agnostic GPU optimizer-step seam: recognized layouts,
//! hyperparameters, the step trait, and the process-global step
//! registry. The interpreter-coupled RECOGNITION (reading models,
//! extracting X/Y from the loss expr) stays above in the evaluator;
//! the GPU compute crates (`mlpl-cuda-eval`, `mlpl-mlx-eval`)
//! implement [`GpuAdamStep`] against only these types.

use std::sync::{Arc, OnceLock};

use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;

use crate::gpu_env::GpuEnv;

/// Param names recovered from the demo `LoRA` model. Everything but
/// the head adapters (`head_a`, `head_b_adapter`) is a frozen base
/// weight.
pub struct DemoLayout {
    pub embed_table: String,
    pub vocab: usize,
    pub wq: String,
    pub wk: String,
    pub wv: String,
    pub wo: String,
    pub head_w: String,
    pub head_b: String,
    pub head_a: String,
    pub head_b_adapter: String,
    pub alpha: f64,
    pub rank: usize,
}

/// One LoRA-wrapped linear layer's param names + scale, as
/// recognized from `Chain[LinearLora, Activation(Relu), LinearLora]`.
pub struct LoraNames {
    pub w: String,
    pub b: String,
    pub a: String,
    pub b_adapter: String,
    pub scale: f32,
}

/// Device-agnostic Adam hyperparameters; each backend converts to
/// its own representation.
#[derive(Clone, Copy, Debug)]
pub struct AdamHp {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub t: i32,
}

/// One GPU optimizer step for a RECOGNIZED architecture. Recognition
/// is done by the caller (`eval_adam`); the impl gets the resolved
/// layout + input tensors + the [`GpuEnv`] accessor, and returns the
/// step loss.
pub trait GpuAdamStep: std::fmt::Debug + Send + Sync {
    /// Head-only `LoRA` fine-tune step.
    ///
    /// # Errors
    /// Backend-specific compute failures, surfaced as `EvalError`.
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError>;
    /// Board-policy MLP step (`Chain[LinearLora, relu, LinearLora]`).
    ///
    /// # Errors
    /// Backend-specific compute failures, surfaced as `EvalError`.
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

/// The step registered by the binary at startup. First write wins;
/// later writes are ignored (idempotent), so tests and the binary
/// can both call the register entry without ordering hazards.
static GPU_STEP: OnceLock<Arc<dyn GpuAdamStep>> = OnceLock::new();

/// Register `step` as this process's GPU optimizer step. Idempotent:
/// only the first registration takes effect.
pub fn register_gpu_step(step: Arc<dyn GpuAdamStep>) {
    let _ = GPU_STEP.set(step);
}

/// This process's installed GPU step, or `None` if the binary never
/// registered one (in which case the optimizer uses the CPU tape).
/// Read by `Environment::new` so every environment in the process
/// shares the one step the binary installed at startup.
#[must_use]
pub fn installed_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    GPU_STEP.get().cloned()
}
