//! Pure pieces of the resident optimizer (saga E4 step 006): the
//! lazy Adam composition, tape seeding with the weight cache, and
//! the per-parameter state writeback.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor, param_from_handle};
use mlpl_tensor_handle::{BinKind, TensorHandle, UnaryKind, device_ops};

use crate::env::Environment;
use crate::env_api::*;

/// Hyperparameters for one resident Adam update.
pub(crate) struct ResidentHp {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub bc1: f64,
    pub bc2: f64,
}

/// A broadcastable resident scalar.
pub(crate) fn scalar(v: f64) -> Option<TensorHandle> {
    Some(TensorHandle::Dev(device_ops()?.full(&[], v).ok()?))
}

/// Resident zeros of `dims`, if the backend is up.
pub(crate) fn fill_like(dims: &[usize]) -> Option<TensorHandle> {
    Some(TensorHandle::Dev(device_ops()?.full(dims, 0.0).ok()?))
}

/// Seed every tracked param onto `tape`, preferring the cached
/// resident weight when its host mirror is unchanged (pointer
/// witness) so steady-state steps re-upload nothing.
pub(crate) fn seed_params(tape: &Rc<Tape>, env: &Environment) -> HashMap<String, Tensor> {
    let mut params = HashMap::new();
    for (name, value) in env.params() {
        let key = ("resident".to_string(), name.clone(), "w".to_string());
        let witness = value.data().as_ptr() as usize;
        let cached = env.optim_state.resident_witness.get(name) == Some(&witness);
        let tensor = match env.optim_state.resident.get(&key) {
            Some(h) if cached => param_from_handle(Rc::clone(tape), h.clone()),
            _ => Tensor::param(Rc::clone(tape), value.clone()),
        };
        params.insert(name.clone(), tensor);
    }
    params
}

/// `m' = b1*m + (1-b1)*g` (also `v'` with `b2` and `g*g`).
fn moment_update(old: &TensorHandle, g: &TensorHandle, beta: f64) -> Option<TensorHandle> {
    old.dev_binary(BinKind::Mul, &scalar(beta)?)
        .ok()?
        .dev_binary(
            BinKind::Add,
            &g.dev_binary(BinKind::Mul, &scalar(1.0 - beta)?).ok()?,
        )
        .ok()
}

/// The lazy Adam update: returns `(m', v', w')` handles.
pub(crate) fn adam_math(
    w: &TensorHandle,
    g: &TensorHandle,
    m: &TensorHandle,
    v: &TensorHandle,
    hp: &ResidentHp,
) -> Option<(TensorHandle, TensorHandle, TensorHandle)> {
    let m_new = moment_update(m, g, hp.b1)?;
    let gg = g.dev_binary(BinKind::Mul, g).ok()?;
    let v_new = moment_update(v, &gg, hp.b2)?;
    let vhat = v_new
        .dev_binary(BinKind::Mul, &scalar(1.0 / hp.bc2)?)
        .ok()?;
    let den = vhat
        .dev_unary(UnaryKind::Sqrt)
        .ok()?
        .dev_binary(BinKind::Add, &scalar(hp.eps)?)
        .ok()?;
    let step = m_new
        .dev_binary(BinKind::Mul, &scalar(hp.lr / hp.bc1)?)
        .ok()?
        .dev_binary(BinKind::Div, &den)
        .ok()?;
    let w_new = w.dev_binary(BinKind::Sub, &step).ok()?;
    Some((m_new, v_new, w_new))
}

/// The lazy momentum update: `v' = beta*v + g`, `w' = w - lr*v'`.
pub(crate) fn momentum_math(
    w: &TensorHandle,
    g: &TensorHandle,
    v_old: &TensorHandle,
    lr: f64,
    beta: f64,
) -> Option<(TensorHandle, TensorHandle)> {
    let v_new = v_old
        .dev_binary(BinKind::Mul, &scalar(beta)?)
        .ok()?
        .dev_binary(BinKind::Add, g)
        .ok()?;
    let step = v_new.dev_binary(BinKind::Mul, &scalar(lr)?).ok()?;
    let w_new = w.dev_binary(BinKind::Sub, &step).ok()?;
    Some((v_new, w_new))
}

/// Write one parameter's post-step state: refresh the host mirror
/// (the ONE download per param per step, so eager readers stay
/// correct), park the new weight + moment handles resident, stamp
/// the cache witness, and drop the superseded host `buffers`
/// entries so a later CPU run restarts cleanly instead of silently
/// using stale moments.
pub(crate) fn commit_update(
    env: &mut Environment,
    name: &str,
    moments: &[((String, String, String), TensorHandle)],
    w_new: TensorHandle,
) {
    let host = w_new.to_dense();
    let witness = host.data().as_ptr() as usize;
    env.set(name.to_string(), host);
    let st = &mut env.optim_state;
    for (key, handle) in moments {
        st.resident.insert(key.clone(), handle.clone());
        st.buffers.remove(key);
    }
    st.resident
        .insert(("resident".into(), name.to_string(), "w".into()), w_new);
    st.resident_witness.insert(name.to_string(), witness);
}
