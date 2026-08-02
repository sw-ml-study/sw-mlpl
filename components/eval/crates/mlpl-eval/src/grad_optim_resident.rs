//! Device-resident optimizer steps (saga E4 step 006). Under
//! `device("mlx")` with a registered backend, `adam` /
//! `momentum_sgd` run ONE tape per step (the CPU generic path used
//! to rebuild a full tape PER PARAMETER), keep the weights and the
//! optimizer moments resident across the whole training loop, and
//! compose the update math from lazy device ops (see the sibling
//! `grad_optim_resident_math`). The host mirror of each weight is
//! refreshed once per step (`to_dense`) so every eager reader
//! (metrics, display, `engram_stats`) stays correct; the moments
//! never leave the device. The gpu_step fast path is untouched.

use std::collections::HashMap;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_tensor_handle::{TensorHandle, device_ops};

use crate::env::Environment;
use crate::env_api::*;
pub(crate) use crate::grad_optim_resident_math::ResidentHp;
use crate::grad_optim_resident_math::{
    adam_math, commit_update, fill_like, momentum_math, seed_params,
};
use mlpl_eval_types::EvalError;

/// The backend vanished mid-step (should not happen: registration
/// is process-global and permanent).
fn backend_lost() -> EvalError {
    EvalError::Unsupported("resident optimizer: device backend unavailable mid-step".into())
}

/// Adam moment keys for `name`.
fn moment_keys(name: &str) -> ((String, String, String), (String, String, String)) {
    (
        ("adam".into(), name.into(), "m".into()),
        ("adam".into(), name.into(), "v".into()),
    )
}

/// Build ONE resident tape for `loss`, backward it, and return each
/// tracked parameter's `(weight, gradient)` handles.
/// Per-param `(resident weight, resident gradient)` pair.
type WgPair = (TensorHandle, TensorHandle);

fn grads_all(
    loss: &Expr,
    env: &mut Environment,
) -> Result<(f64, HashMap<String, WgPair>), EvalError> {
    let tape = mlpl_autograd::Tape::new();
    crate::device::enable_resident_tape(&tape);
    let params = seed_params(&tape, env);
    let root = crate::grad::eval_tensor_expr(loss, env, &tape, &params)?;
    root.backward();
    // One scalar download per step: the allowed reporting sync.
    let loss_val = root.value().data().first().copied().unwrap_or(0.0);
    let nodes = tape.nodes();
    let grads = params
        .into_iter()
        .map(|(n, t)| {
            let node = &nodes[t.node().0];
            let g = node
                .grad
                .clone()
                .unwrap_or_else(|| TensorHandle::Cpu(DenseArray::from_scalar(0.0)));
            (n, (node.value.clone(), g))
        })
        .collect();
    Ok((loss_val, grads))
}

/// One resident Adam step over `names`. `None` = not applicable
/// (wrong device / no backend); the caller falls through to the
/// exact CPU path.
pub(crate) fn try_adam(
    loss: &Expr,
    names: &[String],
    hp: &ResidentHp,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "mlx" {
        return None;
    }
    // The gate must not race the first device-block use: register
    // the backend (idempotent) BEFORE checking, or the process's
    // first train step silently runs the CPU optimizer.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    mlpl_mlx_handle::register_mlx_device_ops();
    device_ops()?;
    Some(adam_steps(loss, names, hp, env))
}

/// The fallible body of [`try_adam`].
fn adam_steps(
    loss: &Expr,
    names: &[String],
    hp: &ResidentHp,
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    let (step_loss, mut grads) = grads_all(loss, env)?;
    for name in names {
        if env.is_frozen(name) {
            continue;
        }
        let (w, g) = grads.remove(name).ok_or_else(|| {
            EvalError::Unsupported(format!("adam: '{name}' is not a tracked parameter"))
        })?;
        adam_one(env, name, &w, &g, hp)?;
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

/// One parameter's resident Adam update: fetch moments (resident
/// slot, else IMPORT the host buffer a CPU phase wrote -- the
/// cpu->device continuity contract -- else zeros), compose the
/// lazy math, commit.
fn adam_one(
    env: &mut Environment,
    name: &str,
    w: &TensorHandle,
    g: &TensorHandle,
    hp: &ResidentHp,
) -> Result<(), EvalError> {
    let (m_key, v_key) = moment_keys(name);
    let fetch = |st: &Environment, key: &(String, String, String)| {
        st.optim_state
            .resident
            .get(key)
            .cloned()
            .or_else(|| mlpl_tensor_handle::upload(st.optim_state.buffers.get(key)?).ok())
    };
    let m_old = fetch(env, &m_key)
        .or_else(|| fill_like(&w.dims()))
        .ok_or_else(backend_lost)?;
    let v_old = fetch(env, &v_key)
        .or_else(|| fill_like(&w.dims()))
        .ok_or_else(backend_lost)?;
    let (m_new, v_new, w_new) = adam_math(w, g, &m_old, &v_old, hp).ok_or_else(backend_lost)?;
    commit_update(env, name, &[(m_key, m_new), (v_key, v_new)], w_new);
    Ok(())
}

/// One resident momentum-SGD step (`v' = beta*v + g`,
/// `w' = w - lr*v'`). Same applicability contract as [`try_adam`].
pub(crate) fn try_momentum(
    loss: &Expr,
    names: &[String],
    lr: f64,
    beta: f64,
    env: &mut Environment,
) -> Option<Result<DenseArray, EvalError>> {
    if env.device() != "mlx" {
        return None;
    }
    // Same first-use registration contract as `try_adam`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    mlpl_mlx_handle::register_mlx_device_ops();
    device_ops()?;
    Some(momentum_steps(loss, names, lr, beta, env))
}

/// The fallible body of [`try_momentum`].
fn momentum_steps(
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
