//! The on-device fine-tune step for the tic-tac-toe board-policy MLP on
//! CUDA: forward + backward + Adam over the four LoRA adapters on the
//! NVIDIA GPU, frozen bases, moments persisted in `env.optim_state`. The
//! CUDA analog of `grad_optim_mlx_mlp_step`; the model recognizer
//! (`mlp_layout`) is device-agnostic and shared with the MLX path.

use crate::env::Environment;
use crate::grad_optim_cuda_step::{step_adapter, tokens_cuda};
use crate::grad_optim_mlx_demo::extract_xy;
use crate::grad_optim_mlx_mlp::{LoraNames, mlp_layout};
use candle_core::Tensor;
use mlpl_array::DenseArray;
use mlpl_cuda_model::{MlpWeights, mlp_forward};
use mlpl_cuda_rt::dense_to_cuda;
use mlpl_cuda_train::{AdamHp, loss_and_grads};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

/// CUDA fast path for one fine-tune step of the board-policy MLP. Returns
/// `None` (CPU fallback) off-`device("cuda")` or for any other model
/// shape (`Chain[LinearLora, relu, LinearLora]` is the only match).
pub(crate) fn try_mlp_adam(
    loss: &Expr,
    model_arg: &Expr,
    hp: &AdamHp,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "cuda" {
        return None;
    }
    let (l1, head) = mlp_layout(model_arg, env)?;
    let (xt, yt) = extract_xy(loss, env)?;
    Some(run_step(&l1, &head, &xt, &yt, hp, env))
}

fn pull(env: &Environment, n: &str) -> Tensor {
    let d = env.get(n).expect("param present");
    dense_to_cuda(d.data(), d.shape().dims())
}

/// Assemble the frozen base weights as candle tensors.
fn mlp_weights(l1: &LoraNames, head: &LoraNames, env: &Environment) -> MlpWeights {
    MlpWeights {
        w1: pull(env, &l1.w),
        b1: pull(env, &l1.b),
        w2: pull(env, &head.w),
        b2: pull(env, &head.b),
        scale1: f64::from(l1.scale),
        scale2: f64::from(head.scale),
    }
}

fn run_step(
    l1: &LoraNames,
    head: &LoraNames,
    xt: &DenseArray,
    yt: &DenseArray,
    hp: &AdamHp,
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    let classes = env.get(&head.w).map_or(0, |a| a.shape().dims()[1]);
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
