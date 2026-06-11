//! Step 010: true-GPU LoRA fine-tune step. When `device("mlx")` and the
//! model is the demo architecture (head-only LoRA), run one fine-tune
//! step's forward + backward + optimizer on the Apple GPU --
//! `mlpl-mlx-model::demo_forward` differentiated by `value_and_grad`,
//! updated by a stateless MLX adam -- instead of the CPU autograd tape.
//! Optimizer moments persist across steps in `env.optim_state` (same as
//! the CPU path), since the interpreter calls `adam(...)` once per step.
//! The per-step plumbing lives in `grad_optim_mlx_step`.

use crate::env::Environment;
use crate::grad_optim_mlx_demo::{DemoLayout, demo_layout, extract_xy};
use crate::grad_optim_mlx_step::{build_weights, step_adapter, tokens_mlx};
use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;
use mlpl_mlx_model::demo_forward;
use mlpl_mlx_rt::dense_to_mlx;
use mlpl_mlx_train::{AdamHp, loss_and_grads};
use mlpl_parser::Expr;

/// MLX fast path for a head-only LoRA fine-tune step. Returns `None` to
/// fall back to the CPU tape adam (non-mlx device, or a model that is
/// not the recognized demo architecture).
pub(crate) fn try_lora_adam(
    loss: &Expr,
    model_arg: &Expr,
    hp: &AdamHp,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "mlx" {
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
    let x_oh = tokens_mlx(xt, layout.vocab)?;
    let y_oh = tokens_mlx(yt, layout.vocab)?;
    let w = build_weights(layout, env, xt.data().len())?;
    let pull = |n: &str| {
        let d = env.get(n).expect("adapter present");
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
    fn try_lora_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &crate::gpu_step::AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>> {
        try_lora_adam(loss, model, &to_mlx_hp(hp), env)
    }

    fn try_mlp_adam(
        &self,
        loss: &Expr,
        model: &Expr,
        hp: &crate::gpu_step::AdamHp,
        env: &mut Environment,
    ) -> Option<Result<DenseArray, EvalError>> {
        crate::grad_optim_mlx_mlp_step::try_mlp_adam(loss, model, &to_mlx_hp(hp), env)
    }
}
