//! True-GPU LoRA fine-tune step on MLX. Runs one fine-tune step's
//! forward + backward + optimizer for the head-only LoRA architecture on
//! the Apple GPU -- `mlpl-mlx-model::demo_forward` differentiated by
//! `value_and_grad` (`loss_and_grads`), updated by a stateless MLX adam
//! -- instead of the CPU autograd tape. Optimizer moments persist across
//! steps via the [`mlpl_eval::GpuEnv`] accessor. Architecture RECOGNITION
//! is interpreter-coupled and lives in `mlpl-eval::eval_adam`; this module
//! gets the resolved `DemoLayout` + input tensors. The CUDA analog lives
//! in `mlpl-cuda-eval::cuda_lora`.

use std::sync::Arc;

use mlpl_array::DenseArray;
use mlpl_eval::{AdamHp as SeamHp, DemoLayout, GpuAdamStep, GpuEnv, LoraNames};
use mlpl_eval_types::EvalError;
use mlpl_mlx_model::demo_forward;
use mlpl_mlx_rt::dense_to_mlx;
use mlpl_mlx_train::{AdamHp, loss_and_grads};

use crate::mlx_step::{build_weights, step_adapter, tokens_mlx};

/// Convert the device-agnostic seam hyperparameters to MLX's (f32).
fn to_mlx_hp(hp: &SeamHp) -> AdamHp {
    AdamHp {
        lr: hp.lr as f32,
        b1: hp.b1 as f32,
        b2: hp.b2 as f32,
        eps: hp.eps as f32,
        t: hp.t,
    }
}

/// The MLX [`GpuAdamStep`](mlpl_eval::GpuAdamStep). The binary registers
/// it with `mlpl_eval::register_gpu_step` at startup; `Environment::new`
/// then hands it to `eval_adam` for `device("mlx") { }` blocks.
#[derive(Debug)]
struct MlxGpuAdam;

impl GpuAdamStep for MlxGpuAdam {
    /// One head-only LoRA fine-tune step on the GPU for the resolved layout.
    fn run_lora_step(
        &self,
        layout: &DemoLayout,
        x: &DenseArray,
        y: &DenseArray,
        hp: &SeamHp,
        env: &mut dyn GpuEnv,
    ) -> Result<DenseArray, EvalError> {
        let hp = to_mlx_hp(hp);
        let x_oh = tokens_mlx(x, layout.vocab)?;
        let y_oh = tokens_mlx(y, layout.vocab)?;
        let w = build_weights(layout, env, x.data().len())?;
        let pull = |n: &str| {
            let d = env.binding(n).expect("adapter present");
            dense_to_mlx(d.data(), d.shape().dims())
        };
        let adapters = [pull(&layout.head_a), pull(&layout.head_b_adapter)];
        let (loss, grads) = loss_and_grads(&adapters, |a| {
            demo_forward(&w, a, &x_oh, &y_oh).map(|l| vec![l])
        })
        .map_err(|e| EvalError::Unsupported(format!("mlx lora: {e}")))?;
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
        crate::mlx_mlp::run_step(l1, head, x, y, &to_mlx_hp(hp), env)
    }
}

/// This build's MLX optimizer step, for the binary to register at startup
/// via `mlpl_eval::register_gpu_step`.
#[must_use]
pub fn gpu_step() -> Arc<dyn GpuAdamStep> {
    Arc::new(MlxGpuAdam)
}
