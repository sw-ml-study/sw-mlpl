//! The on-device fine-tune step for the tic-tac-toe board-policy MLP on
//! CUDA: forward + backward + Adam over the four LoRA adapters on the
//! NVIDIA GPU, frozen bases, moments persisted via the `mlpl_eval::GpuEnv`
//! accessor. Recognition (`mlp_layout`) is interpreter-coupled and lives
//! in `mlpl-eval::eval_adam`; this module gets the resolved `LoraNames`
//! layout + input tensors.

use candle_core::Tensor;
use mlpl_array::DenseArray;
use mlpl_cuda_model::{MlpWeights, mlp_forward};
use mlpl_cuda_rt::dense_to_cuda;
use mlpl_cuda_train::{AdamHp, loss_and_grads};
use mlpl_eval::{GpuEnv, LoraNames};
use mlpl_eval_types::EvalError;

use crate::cuda_step::{step_adapter, tokens_cuda};

fn pull(env: &dyn GpuEnv, n: &str) -> Tensor {
    let d = env.binding(n).expect("param present");
    dense_to_cuda(d.data(), d.shape().dims())
}

/// Assemble the frozen base weights as candle tensors.
fn mlp_weights(l1: &LoraNames, head: &LoraNames, env: &dyn GpuEnv) -> MlpWeights {
    MlpWeights {
        w1: pull(env, &l1.w),
        b1: pull(env, &l1.b),
        w2: pull(env, &head.w),
        b2: pull(env, &head.b),
        scale1: f64::from(l1.scale),
        scale2: f64::from(head.scale),
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
    let x = dense_to_cuda(xt.data(), xt.shape().dims());
    let y_oh = tokens_cuda(yt, classes)?;
    let w = mlp_weights(l1, head, env);
    let names = [&l1.a, &l1.b_adapter, &head.a, &head.b_adapter];
    let adapters: Vec<Tensor> = names.iter().map(|n| pull(env, n)).collect();
    let (loss, grads) = loss_and_grads(&adapters, |a| mlp_forward(&w, a, &x, &y_oh))
        .map_err(|e| EvalError::Unsupported(format!("cuda mlp: {e}")))?;
    for (name, grad) in names.iter().zip(grads.iter()) {
        step_adapter(env, name, grad, hp)?;
    }
    Ok(DenseArray::from_scalar(f64::from(loss)))
}
