//! Step 010: true-GPU LoRA fine-tune step. Runs one fine-tune step's
//! forward + backward + optimizer for the head-only LoRA architecture on
//! the Apple GPU -- `mlpl-mlx-model::demo_forward` differentiated by
//! `value_and_grad`, updated by a stateless MLX adam -- instead of the
//! CPU autograd tape. Optimizer moments persist across steps via the
//! `GpuEnv` accessor. Architecture RECOGNITION is interpreter-coupled and
//! lives in `grad_optim::eval_adam`; this module gets the resolved
//! `DemoLayout` + input tensors. The MLX analog of `grad_optim_cuda`.

use crate::gpu_step::GpuEnv;
use crate::grad_optim_mlx_demo::DemoLayout;
use crate::grad_optim_mlx_mlp::LoraNames;
use crate::grad_optim_mlx_step::{build_weights, step_adapter, tokens_mlx};
use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;
use mlpl_mlx_model::demo_forward;
use mlpl_mlx_rt::dense_to_mlx;
use mlpl_mlx_train::{AdamHp, loss_and_grads};

/// One head-only LoRA fine-tune step on the GPU for the resolved layout.
fn run_step(
    layout: &DemoLayout,
    xt: &DenseArray,
    yt: &DenseArray,
    hp: &AdamHp,
    env: &mut dyn GpuEnv,
) -> Result<DenseArray, EvalError> {
    let x_oh = tokens_mlx(xt, layout.vocab)?;
    let y_oh = tokens_mlx(yt, layout.vocab)?;
    let w = build_weights(layout, env, xt.data().len())?;
    let pull = |n: &str| {
        let d = env.binding(n).expect("adapter present");
        dense_to_mlx(d.data(), d.shape().dims())
    };
    let adapters = [pull(&layout.head_a), pull(&layout.head_b_adapter)];
    let (loss, grads) = loss_and_grads(&adapters, |a| {
        demo_forward(&w, a, &x_oh, &y_oh).map(|l| vec![l])
    })
    .map_err(|e| EvalError::Unsupported(format!("mlx lora: {e}")))?;
    step_adapter(env, &layout.head_a, &grads[0], hp)?;
    step_adapter(env, &layout.head_b_adapter, &grads[1], hp)?;
    Ok(DenseArray::from_scalar(f64::from(loss)))
}

/// Convert the device-agnostic hyperparameters to MLX's (f32).
fn to_mlx_hp(hp: &crate::gpu_step::AdamHp) -> AdamHp {
    AdamHp {
        lr: hp.lr as f32,
        b1: hp.b1 as f32,
        b2: hp.b2 as f32,
        eps: hp.eps as f32,
        t: hp.t,
    }
}

/// The MLX [`GpuAdamStep`](crate::gpu_step::GpuAdamStep) -- registered
/// on every `Environment` in an mlx/macos/aarch64 build.
#[derive(Debug)]
pub(crate) struct MlxGpuAdam;

impl crate::gpu_step::GpuAdamStep for MlxGpuAdam {
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &crate::gpu_step::AdamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        run_step(layout, x, y, &to_mlx_hp(hp), env)
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
        crate::grad_optim_mlx_mlp_step::run_step(l1, head, x, y, &to_mlx_hp(hp), env)
    }
}
