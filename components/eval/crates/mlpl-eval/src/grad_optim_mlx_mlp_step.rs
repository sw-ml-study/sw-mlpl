//! The on-device fine-tune step for the tic-tac-toe board-policy MLP:
//! forward + backward + Adam over the four LoRA adapters on the Apple
//! GPU, frozen bases, moments persisted in `env.optim_state`. The model
//! recognizer lives in `grad_optim_mlx_mlp`.

use crate::env::Environment;
use crate::grad_optim_mlx_demo::extract_xy;
use crate::grad_optim_mlx_mlp::{LoraNames, mlp_layout};
use crate::grad_optim_mlx_step::{step_adapter, tokens_mlx};
use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;
use mlpl_mlx_model::{MlpWeights, mlp_forward};
use mlpl_mlx_rt::{Array, dense_to_mlx};
use mlpl_mlx_train::{AdamHp, loss_and_grads};
use mlpl_parser::Expr;

/// MLX fast path for one fine-tune step of the board-policy MLP.
/// Returns `None` (CPU fallback) off-`device("mlx")` or for any other
/// model shape.
pub(crate) fn try_mlp_adam(
    loss: &Expr,
    model_arg: &Expr,
    hp: &AdamHp,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "mlx" {
        return None;
    }
    let (l1, head) = mlp_layout(model_arg, env)?;
    let (xt, yt) = extract_xy(loss, env)?;
    Some(run_step(&l1, &head, &xt, &yt, hp, env))
}

fn pull_array(env: &Environment, n: &str) -> Array {
    let d = env.get(n).expect("param present");
    dense_to_mlx(d.data(), d.shape().dims())
}

/// Assemble the frozen base weights as MLX arrays.
fn mlp_weights(l1: &LoraNames, head: &LoraNames, env: &Environment) -> MlpWeights {
    MlpWeights {
        w1: pull_array(env, &l1.w),
        b1: pull_array(env, &l1.b),
        w2: pull_array(env, &head.w),
        b2: pull_array(env, &head.b),
        scale1: l1.scale,
        scale2: head.scale,
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
    let x = dense_to_mlx(xt.data(), xt.shape().dims());
    let y_oh = tokens_mlx(yt, classes)?;
    let w = mlp_weights(l1, head, env);
    let names = [&l1.a, &l1.b_adapter, &head.a, &head.b_adapter];
    let adapters: Vec<Array> = names.iter().map(|n| pull_array(env, n)).collect();
    let (loss, grads) = loss_and_grads(&adapters, |a| {
        mlp_forward(&w, a, &x, &y_oh).map(|l| vec![l])
    })
    .map_err(|e| EvalError::Unsupported(format!("mlx mlp: {e}")))?;
    for (name, grad) in names.iter().zip(grads.iter()) {
        step_adapter(env, name, grad, hp)?;
    }
    Ok(DenseArray::from_scalar(f64::from(loss)))
}
