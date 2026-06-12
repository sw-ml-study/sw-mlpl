//! Process-global registration of this build's GPU optimizer step.
//!
//! Stage 2 of the GPU workspace split (docs/build-and-workspace-plan.md):
//! the cycle-break. `Environment::new` no longer NAMES the concrete
//! `GpuAdamStep` impl -- it reads whatever was registered here. The
//! binary (`mlpl-serve` / `mlpl-repl`, feature-gated) registers the right
//! step once at startup.
//!
//! Post-S3: the CUDA step lives in the sibling `mlpl-cuda-eval` crate, so
//! the binary registers `mlpl_cuda_eval::gpu_step()` via
//! [`register_gpu_step`] and `default_gpu_step()` returns `None` on CUDA.
//! The MLX step is still in-crate, so the binary still uses the no-arg
//! [`register_default_gpu_step`] there. S4 moves MLX out too, after which
//! `default_gpu_step` (and the fallback below) can be deleted and
//! registration becomes mandatory.

use std::sync::{Arc, OnceLock};

use crate::gpu_step::{GpuAdamStep, default_gpu_step};

/// The step registered by the binary at startup. First write wins;
/// later writes are ignored (idempotent), so tests and the binary can
/// both call the register entry without ordering hazards.
static GPU_STEP: OnceLock<Arc<dyn GpuAdamStep>> = OnceLock::new();

/// Register `step` as this process's GPU optimizer step. Idempotent:
/// only the first registration takes effect.
pub fn register_gpu_step(step: Arc<dyn GpuAdamStep>) {
    let _ = GPU_STEP.set(step);
}

/// Register this build's default GPU step (CUDA on linux/x86_64, MLX on
/// macos/aarch64). The binary calls this once at startup. No-arg so the
/// caller never has to name the `GpuAdamStep` trait; in S3/S4 the binary
/// switches to registering the sibling crate's step directly.
pub fn register_default_gpu_step() {
    if let Some(step) = default_gpu_step() {
        register_gpu_step(step);
    }
}

/// This process's installed GPU step: the registered one if a binary set
/// it, else the in-crate default. The fallback keeps `Environment::new`
/// working for callers that don't register -- the MLX in-crate GPU demo
/// tests, and the REPL before its register call. On CUDA the default is
/// `None` (the step moved to `mlpl-cuda-eval`); the fallback is deleted in
/// S4 once MLX moves out too and registration becomes mandatory.
pub(crate) fn installed_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    GPU_STEP.get().cloned().or_else(default_gpu_step)
}
