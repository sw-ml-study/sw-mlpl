//! Step 006: true-GPU LoRA fine-tune step on CUDA. When
//! `device("cuda")` and the model is the demo architecture (head-only
//! LoRA), run one fine-tune step's forward + backward + optimizer on
//! the NVIDIA GPU -- `mlpl-cuda-model::demo_forward` differentiated by
//! candle autograd (`loss_and_grads`), updated by a stateless candle
//! adam -- instead of the CPU autograd tape. Optimizer moments persist
//! across steps in `env.optim_state` (same as the CPU/MLX paths). The
//! CUDA analog of `grad_optim_mlx`; the demo-architecture recognition
//! is shared (`grad_optim_mlx_demo`).

use crate::env::Environment;
use crate::grad_optim_cuda_step::{build_weights, step_adapter, tokens_cuda};
use crate::grad_optim_mlx_demo::{DemoLayout, demo_layout, extract_xy};
use mlpl_array::DenseArray;
use mlpl_cuda_model::demo_forward;
use mlpl_cuda_rt::dense_to_cuda;
use mlpl_cuda_train::{AdamHp, loss_and_grads};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

/// CUDA fast path for a head-only LoRA fine-tune step. Returns `None`
/// to fall back to the CPU tape adam (non-cuda device, or a model that
/// is not the recognized demo architecture).
pub(crate) fn try_lora_adam(
    loss: &Expr,
    model_arg: &Expr,
    hp: &AdamHp,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "cuda" {
        return None;
    }
    let layout = demo_layout(model_arg, env)?;
    let (xt, yt) = extract_xy(loss, env)?;
    Some(run_step(&layout, &xt, &yt, hp, env))
}

fn run_step(
    layout: &DemoLayout,
    xt: &DenseArray,
    yt: &DenseArray,
    hp: &AdamHp,
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    let x_oh = tokens_cuda(xt, layout.vocab)?;
    let y_oh = tokens_cuda(yt, layout.vocab)?;
    let w = build_weights(layout, env, xt.data().len())?;
    let pull = |n: &str| {
        let d = env.get(n).expect("adapter present");
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
    fn try_lora_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &crate::gpu_step::AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>> {
        try_lora_adam(loss, model, &to_cuda_hp(hp), env)
    }

    fn try_mlp_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &crate::gpu_step::AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>> {
        crate::grad_optim_cuda_mlp::try_mlp_adam(loss, model, &to_cuda_hp(hp), env)
    }
}
