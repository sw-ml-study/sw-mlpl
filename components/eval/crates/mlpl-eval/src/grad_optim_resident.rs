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
use crate::grad_optim_resident_adam::adam_steps;
pub(crate) use crate::grad_optim_resident_math::ResidentHp;
use crate::grad_optim_resident_math::seed_params;
use crate::grad_optim_resident_momentum::momentum_steps;
use mlpl_eval_types::EvalError;

/// The backend vanished mid-step (should not happen: registration
/// is process-global and permanent). Shared by both optimizer steps.
pub(crate) fn backend_lost() -> EvalError {
    EvalError::Unsupported("resident optimizer: device backend unavailable mid-step".into())
}

/// Build ONE resident tape for `loss`, backward it, and return each
/// tracked parameter's `(weight, gradient)` handles.
/// Per-param `(resident weight, resident gradient)` pair.
type WgPair = (TensorHandle, TensorHandle);

pub(crate) fn grads_all(
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
