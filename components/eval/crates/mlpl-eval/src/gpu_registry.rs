//! Process-global registration of this build's GPU optimizer step.
//!
//! Stage 2 of the GPU workspace split (docs/build-and-workspace-plan.md):
//! the cycle-break. `Environment::new` no longer NAMES the concrete
//! `GpuAdamStep` impl -- it reads whatever was registered here. The
//! binary (`mlpl-serve` / `mlpl-repl`, feature-gated) registers the right
//! step once at startup via [`register_default_gpu_step`]. Today the
//! default is still constructed inside this crate (`default_gpu_step`);
//! in S3/S4 that constructor moves to the sibling `mlpl-cuda-eval` /
//! `mlpl-mlx-eval` crates and the binary registers THEIR step instead --
//! at which point the in-crate fallback below is deleted and registration
//! becomes mandatory.

use std::sync::{Arc, OnceLock};

use crate::gpu_step::{GpuAdamStep, default_gpu_step};

/// The step registered by the binary at startup. First write wins;
/// later writes are ignored (idempotent), so tests and the binary can
/// both call the register entry without ordering hazards.
static GPU_STEP: OnceLock<Arc<dyn GpuAdamStep>> = OnceLock::new();

/// Register `step` as this process's GPU optimizer step. Idempotent:
/// only the first registration takes effect.
pub(crate) fn register_gpu_step(step: Arc<dyn GpuAdamStep>) {
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
/// working for callers that don't register (the in-crate GPU tests, the
/// REPL before its register call) and is removed in S3 when the concrete
/// impl leaves this crate.
pub(crate) fn installed_gpu_step() -> Option<Arc<dyn GpuAdamStep>> {
    GPU_STEP.get().cloned().or_else(default_gpu_step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{AdamHp, GpuEnv};
    use crate::grad_optim_mlx_demo::DemoLayout;
    use crate::grad_optim_mlx_mlp::LoraNames;
    use mlpl_array::DenseArray;
    use mlpl_eval_types::EvalError;

    #[derive(Debug)]
    struct Dummy;
    impl GpuAdamStep for Dummy {
        fn run_lora_step(
            &self,
            _l: &DemoLayout,
            _x: &DenseArray,
            _y: &DenseArray,
            _hp: &AdamHp,
            _e: &mut dyn GpuEnv,
        ) -> Result<DenseArray, EvalError> {
            unreachable!("test never runs the step")
        }
        fn run_mlp_step(
            &self,
            _l1: &LoraNames,
            _h: &LoraNames,
            _x: &DenseArray,
            _y: &DenseArray,
            _hp: &AdamHp,
            _e: &mut dyn GpuEnv,
        ) -> Result<DenseArray, EvalError> {
            unreachable!("test never runs the step")
        }
    }

    #[test]
    fn registered_step_is_what_new_environments_see() {
        let dummy: Arc<dyn GpuAdamStep> = Arc::new(Dummy);
        register_gpu_step(dummy.clone());
        let got = installed_gpu_step().expect("a step is installed");
        assert!(
            Arc::ptr_eq(&got, &dummy),
            "registered step must win over the in-crate fallback"
        );
        // Idempotent: a second register does not replace the first.
        register_gpu_step(Arc::new(Dummy));
        let again = installed_gpu_step().expect("still installed");
        assert!(Arc::ptr_eq(&again, &dummy), "first registration wins");
    }
}
