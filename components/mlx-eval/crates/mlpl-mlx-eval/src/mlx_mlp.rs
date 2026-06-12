//! The on-device fine-tune step for the tic-tac-toe board-policy MLP:
//! forward + backward + Adam over the four LoRA adapters on the Apple
//! GPU, frozen bases, moments persisted via the `GpuEnv` accessor. The
//! MLX analog of mlpl-cuda-eval::cuda_mlp. Recognition (`mlp_layout`) is
//! interpreter-coupled and lives in `eval_adam`; this module gets the
//! resolved `LoraNames` layout + input tensors.

use crate::mlx_step::{step_adapter, tokens_mlx};
use mlpl_array::DenseArray;
use mlpl_eval::{GpuEnv, LoraNames};
use mlpl_eval_types::EvalError;
use mlpl_mlx_model::{MlpWeights, mlp_forward};
use mlpl_mlx_rt::{Array, dense_to_mlx};
use mlpl_mlx_train::{AdamHp, loss_and_grads};

fn pull(env: &dyn GpuEnv, n: &str) -> Array {
    let d = env.binding(n).expect("param present");
    dense_to_mlx(d.data(), d.shape().dims())
}

/// Assemble the frozen base weights as MLX arrays.
fn mlp_weights(l1: &LoraNames, head: &LoraNames, env: &dyn GpuEnv) -> MlpWeights {
    MlpWeights {
        w1: pull(env, &l1.w),
        b1: pull(env, &l1.b),
        w2: pull(env, &head.w),
        b2: pull(env, &head.b),
        scale1: l1.scale,
        scale2: head.scale,
    }
}

/// One board-policy MLP fine-tune step on the GPU for the resolved layout.
pub(crate) fn run_step(
    l1: &LoraNames,
    head: &LoraNames,
    xt: &DenseArray,
    yt: &DenseArray,
    hp: &AdamHp,
    env: &mut dyn GpuEnv,
) -> Result<DenseArray, EvalError> {
    let classes = env.binding(&head.w).map_or(0, |a| a.shape().dims()[1]);
    let x = dense_to_mlx(xt.data(), xt.shape().dims());
    let y_oh = tokens_mlx(yt, classes)?;
    let w = mlp_weights(l1, head, env);
    let names = [&l1.a, &l1.b_adapter, &head.a, &head.b_adapter];
    let adapters: Vec<Array> = names.iter().map(|n| pull(env, n)).collect();
    let (loss, grads) = loss_and_grads(&adapters, |a| {
        mlp_forward(&w, a, &x, &y_oh).map(|l| vec![l])
    })
    .map_err(|e| EvalError::Unsupported(format!("mlx mlp: {e}")))?;
    for (name, grad) in names.iter().zip(grads.iter()) {
        step_adapter(env, name, grad, hp)?;
    }
    Ok(DenseArray::from_scalar(f64::from(loss)))
}
