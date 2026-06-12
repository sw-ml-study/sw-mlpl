//! True-GPU LoRA fine-tune step on CUDA. Runs one fine-tune step's
//! forward + backward + optimizer for the head-only LoRA architecture on
//! the NVIDIA GPU -- `mlpl-cuda-model::demo_forward` differentiated by
//! candle autograd (`loss_and_grads`), updated by a stateless candle adam
//! -- instead of the CPU autograd tape. Optimizer moments persist across
//! steps via the [`mlpl_eval::GpuEnv`] accessor. Architecture RECOGNITION
//! is interpreter-coupled and lives in `mlpl-eval::eval_adam`; this module
//! gets the resolved `DemoLayout` + input tensors.

use std::sync::Arc;

use mlpl_array::DenseArray;
use mlpl_cuda_model::demo_forward;
use mlpl_cuda_rt::dense_to_cuda;
use mlpl_cuda_train::{AdamHp, loss_and_grads};
use mlpl_eval::{AdamHp as SeamHp, DemoLayout, GpuAdamStep, GpuEnv, LoraNames};
use mlpl_eval_types::EvalError;

use crate::cuda_step::{build_weights, step_adapter, tokens_cuda};

/// Convert the device-agnostic seam hyperparameters to candle's.
fn to_cuda_hp(hp: &SeamHp) -> AdamHp {
    AdamHp {
        lr: hp.lr,
        b1: hp.b1,
        b2: hp.b2,
        eps: hp.eps,
        t: hp.t,
    }
}

/// The CUDA [`GpuAdamStep`](mlpl_eval::GpuAdamStep). The binary registers
/// it with `mlpl_eval::register_gpu_step` at startup; `Environment::new`
/// then hands it to `eval_adam` for `device("cuda") { }` blocks.
#[derive(Debug)]
struct CudaGpuAdam;

impl GpuAdamStep for CudaGpuAdam {
    /// One head-only LoRA fine-tune step on the GPU for the resolved layout.
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &SeamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        let hp = to_cuda_hp(hp);
        let x_oh = tokens_cuda(x, layout.vocab)?;
        let y_oh = tokens_cuda(y, layout.vocab)?;
        let w = build_weights(layout, env, x.data().len())?;
        let pull = |n: &str| {
            let d = env.binding(n).expect("adapter present");
            dense_to_cuda(d.data(), d.shape().dims())
        };
        let adapters = [pull(&layout.head_a), pull(&layout.head_b_adapter)];
        let (loss, grads) = loss_and_grads(&adapters, |a| demo_forward(&w, a, &x_oh, &y_oh))
            .map_err(|e| EvalError::Unsupported(format!("cuda lora: {e}")))?;
        step_adapter(env, &layout.head_a, &grads[0], &hp)?;
        step_adapter(env, &layout.head_b_adapter, &grads[1], &hp)?;
        Ok(DenseArray::from_scalar(f64::from(loss)))
    }

    fn run_mlp_step(
        &self,
        l1: &LoraNames,
        head: &LoraNames,
        x: &DenseArray,
        y: &DenseArray,
        hp: &SeamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        crate::cuda_mlp::run_step(l1, head, x, y, &to_cuda_hp(hp), env)
    }
}

/// This build's CUDA optimizer step, for the binary to register at
/// startup via `mlpl_eval::register_gpu_step`.
#[must_use]
pub fn gpu_step() -> Arc<dyn GpuAdamStep> {
    Arc::new(CudaGpuAdam)
}
