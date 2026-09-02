//! The resident momentum-SGD step, split out of `grad_optim_resident`
//! so each optimizer's step body is a small module. Shared plumbing
//! (`grads_all`, `backend_lost`) stays in the parent; the update math is
//! in `grad_optim_resident_math`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_tensor_handle::TensorHandle;

use crate::env::Environment;
use crate::env_api::*;
use crate::grad_optim_resident::{backend_lost, grads_all};
use crate::grad_optim_resident_math::{commit_update, fill_like, momentum_math};
use mlpl_eval_types::EvalError;

/// The fallible body of `try_momentum`.
pub(crate) fn momentum_steps(
    loss: &Expr,
    names: &[String],
    lr: f64,
    beta: f64,
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    let (step_loss, mut grads) = grads_all(loss, env)?;
    for name in names {
        if env.is_frozen(name) {
            continue;
        }
        let (w, g) = grads.remove(name).ok_or_else(|| {
            EvalError::Unsupported(format!("momentum_sgd: '{name}' is not tracked"))
        })?;
        momentum_one(env, name, &w, &g, lr, beta)?;
    }
    Ok(DenseArray::from_scalar(step_loss))
}

/// One parameter's resident momentum update.
fn momentum_one(
    env: &mut Environment,
    name: &str,
    w: &TensorHandle,
    g: &TensorHandle,
    lr: f64,
    beta: f64,
) -> Result<(), EvalError> {
    let v_key = (
        "momentum_sgd".to_string(),
        name.to_string(),
        "v".to_string(),
    );
    let v_old = env
        .optim_state
        .resident
        .get(&v_key)
        .cloned()
        .or_else(|| mlpl_tensor_handle::upload(env.optim_state.buffers.get(&v_key)?).ok())
        .or_else(|| fill_like(&w.dims()))
        .ok_or_else(backend_lost)?;
    let (v_new, w_new) = momentum_math(w, g, &v_old, lr, beta).ok_or_else(backend_lost)?;
    commit_update(env, name, &[(v_key, v_new)], w_new);
    Ok(())
}
