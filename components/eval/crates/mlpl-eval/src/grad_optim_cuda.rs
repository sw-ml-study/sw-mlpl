//! Step 006: true-GPU LoRA fine-tune step on CUDA. Runs one fine-tune
//! step's forward + backward + optimizer for the head-only LoRA
//! architecture on the NVIDIA GPU -- `mlpl-cuda-model::demo_forward`
//! differentiated by candle autograd (`loss_and_grads`), updated by a
//! stateless candle adam -- instead of the CPU autograd tape. Optimizer
//! moments persist across steps via the `GpuEnv` accessor. Architecture
//! RECOGNITION is interpreter-coupled and lives in `grad_optim::eval_adam`;
//! this module gets the resolved `DemoLayout` + input tensors. The CUDA
//! analog of `grad_optim_mlx`.

use crate::gpu_step::GpuEnv;
use crate::grad_optim_cuda_step::{build_weights, step_adapter, tokens_cuda};
use crate::grad_optim_mlx_demo::DemoLayout;
use crate::grad_optim_mlx_mlp::LoraNames;
use mlpl_array::DenseArray;
use mlpl_cuda_model::demo_forward;
use mlpl_cuda_rt::dense_to_cuda;
use mlpl_cuda_train::{AdamHp, loss_and_grads};
use mlpl_eval_types::EvalError;

/// One head-only LoRA fine-tune step on the GPU for the resolved layout.
fn run_step(
    layout: &DemoLayout,
    xt: &DenseArray,
    yt: &DenseArray,
    hp: &AdamHp,
    env: &mut dyn GpuEnv,
) -> Result<DenseArray, EvalError> {
    let x_oh = tokens_cuda(xt, layout.vocab)?;
    let y_oh = tokens_cuda(yt, layout.vocab)?;
    let w = build_weights(layout, env, xt.data().len())?;
    let pull = |n: &str| {
        let d = env.binding(n).expect("adapter present");
        dense_to_cuda(d.data(), d.shape().dims())
    };
    let adapters = [pull(&layout.head_a), pull(&layout.head_b_adapter)];
    let (loss, grads) = loss_and_grads(&adapters, |a| demo_forward(&w, a, &x_oh, &y_oh))
        .map_err(|e| EvalError::Unsupported(format!("cuda lora: {e}")))?;
    step_adapter(env, &layout.head_a, &grads[0], hp)?;
    step_adapter(env, &layout.head_b_adapter, &grads[1], hp)?;
    Ok(DenseArray::from_scalar(f64::from(loss)))
}

/// Convert the device-agnostic hyperparameters to candle's.
fn to_cuda_hp(hp: &crate::gpu_step::AdamHp) -> AdamHp {
    AdamHp {
        lr: hp.lr,
        b1: hp.b1,
        b2: hp.b2,
        eps: hp.eps,
        t: hp.t,
    }
}

/// The CUDA [`GpuAdamStep`](crate::gpu_step::GpuAdamStep) -- registered
/// on every `Environment` in a cuda/linux/x86_64 build.
#[derive(Debug)]
pub(crate) struct CudaGpuAdam;

impl crate::gpu_step::GpuAdamStep for CudaGpuAdam {
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &crate::gpu_step::AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        run_step(layout, x, y, &to_cuda_hp(hp), env)
    }

    fn run_mlp_step(
        &self,
        l1: &LoraNames,
        head: &LoraNames,
        x: &DenseArray,
        y: &DenseArray,
        hp: &crate::gpu_step::AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        crate::grad_optim_cuda_mlp::run_step(l1, head, x, y, &to_cuda_hp(hp), env)
    }
}
