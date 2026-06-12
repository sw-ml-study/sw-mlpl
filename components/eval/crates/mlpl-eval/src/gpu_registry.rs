//! Process-global registration of this build's GPU optimizer step.
//!
//! Stage 2 of the GPU workspace split (docs/build-and-workspace-plan.md):
//! the cycle-break. `Environment::new` no longer NAMES the concrete
//! `GpuAdamStep` impl -- it reads whatever was registered here. The
//! binary (`mlpl-serve` / `mlpl-repl`, feature-gated) registers the right
//! step once at startup.
//!
//! Post-S4: BOTH compute impls live in sibling crates -- the CUDA step in
//! `mlpl-cuda-eval` (S3), the MLX step in `mlpl-mlx-eval` (S4) -- so this
//! crate constructs none of them and there is no in-crate default.
//! Registration is mandatory: the binary registers
//! `mlpl_cuda_eval::gpu_step()` / `mlpl_mlx_eval::gpu_step()` via
//! [`register_gpu_step`], and the moved demo tests register their own
//! crate's step. An unregistered GPU build simply has no GPU fast path --
//! `eval_adam` falls back to the CPU tape with a one-time notice.

use std::sync::{Arc, OnceLock};

use crate::gpu_step::GpuAdamStep;

/// The step registered by the binary at startup. First write wins;
/// later writes are ignored (idempotent), so tests and the binary can
/// both call the register entry without ordering hazards.
static GPU_STEP: OnceLock<Arc<dyn GpuAdamStep>> = OnceLock::new();

/// Register `step` as this process's GPU optimizer step. Idempotent:
/// only the first registration takes effect.
pub fn register_gpu_step(step: Arc<dyn GpuAdamStep>) {
    let _ = GPU_STEP.set(step);
}

/// This process's installed GPU step, or `None` if the binary never
/// registered one (in which case `eval_adam` uses the CPU tape). Read by
/// `Environment::new` so every environment in the process shares the one
/// step the binary installed at startup.
pub(crate) fn installed_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    GPU_STEP.get().cloned()
}
