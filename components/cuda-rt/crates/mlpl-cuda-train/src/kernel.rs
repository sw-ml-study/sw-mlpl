//! The CUDA training kernel: candle autodiff (`loss_and_grads`) and a
//! stateless Adam update (`adam_update`) whose moment buffers are
//! candle `Tensor`s. The CUDA analog of `mlpl-mlx-train`'s kernel +
//! `adam_update`; paired with the interpreter's per-step CUDA `LoRA` path
//! (which persists m/v in the eval Environment between `adam(...)`
//! calls), a fine-tune step runs forward + backward + optimizer on the
//! GPU.

use candle_core::{Error, Result, Tensor, Var};

/// Adam hyperparameters + the 1-based step counter `t`. Bundled so
/// callers pass one value instead of five.
pub struct AdamHp {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay (beta1).
    pub b1: f64,
    /// Second-moment decay (beta2).
    pub b2: f64,
    /// Numerical-stability epsilon.
    pub eps: f64,
    /// 1-based step counter (for bias correction).
    pub t: i32,
}

/// Evaluate `loss_fn(adapters)` and its gradient w.r.t. every adapter,
/// on-device. The adapter tensors are wrapped in candle `Var`s so
/// `backward` can recover their gradients. Returns the scalar loss as
/// `f32` plus one gradient `Tensor` per adapter (same order).
///
/// # Errors
/// Propagates candle errors from the forward, the backward pass, or a
/// missing gradient (an adapter that the loss does not depend on).
pub fn loss_and_grads<F>(adapters: &[Tensor], loss_fn: F) -> Result<(f32, Vec<Tensor>)>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let vars = adapters
        .iter()
        .map(Var::from_tensor)
        .collect::<Result<Vec<_>>>()?;
    let inputs = vars
        .iter()
        .map(|v| v.as_tensor().clone())
        .collect::<Vec<_>>();
    let loss = loss_fn(&inputs)?;
    let store = loss.backward()?;
    let grads = vars
        .iter()
        .map(|v| {
            store
                .get(v)
                .cloned()
                .ok_or_else(|| Error::Msg("adapter has no gradient".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((loss.to_scalar::<f32>()?, grads))
}

/// One stateless Adam update for a single parameter, returning
/// `(w_new, m_new, v_new)`. Moment buffers are passed in and out so the
/// caller can persist them externally (the interpreter keeps m/v in the
/// eval Environment between `adam(...)` calls). All arithmetic is candle
/// ops on the GPU.
///
/// # Errors
/// Propagates candle elementwise-op errors.
pub fn adam_update(
    w: &Tensor,
    g: &Tensor,
    m: &Tensor,
    v: &Tensor,
    hp: &AdamHp,
) -> Result<(Tensor, Tensor, Tensor)> {
    let m_new = m.affine(hp.b1, 0.0)?.add(&g.affine(1.0 - hp.b1, 0.0)?)?;
    let v_new = v
        .affine(hp.b2, 0.0)?
        .add(&g.sqr()?.affine(1.0 - hp.b2, 0.0)?)?;
    let mhat = m_new.affine(1.0 / (1.0 - hp.b1.powi(hp.t)), 0.0)?;
    let vhat = v_new.affine(1.0 / (1.0 - hp.b2.powi(hp.t)), 0.0)?;
    let denom = vhat.sqrt()?.affine(1.0, hp.eps)?;
    let w_new = w.sub(&mhat.div(&denom)?.affine(hp.lr, 0.0)?)?;
    Ok((w_new, m_new, v_new))
}
