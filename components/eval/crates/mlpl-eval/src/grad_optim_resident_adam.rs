//! The resident Adam step, split out of `grad_optim_resident` so each
//! optimizer's step body is a small module. Shared plumbing
//! (`grads_all`, `backend_lost`) stays in the parent; the update math is
//! in `grad_optim_resident_math`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_tensor_handle::TensorHandle;

use crate::env::Environment;
use crate::env_api::*;
use crate::grad_optim_resident::{backend_lost, grads_all};
use crate::grad_optim_resident_math::{ResidentHp, adam_math, commit_update, fill_like};
use mlpl_eval_types::EvalError;

/// Adam moment keys for `name`.
fn moment_keys(name: &str) -> ((String, String, String), (String, String, String)) {
    (
        ("adam".into(), name.into(), "m".into()),
        ("adam".into(), name.into(), "v".into()),
    )
}

/// The fallible body of `try_adam`.
pub(crate) fn adam_steps(
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
