//! The on-device MLX training kernel: a traceable forward primitive,
//! the `value_and_grad` gradient wrapper, and a step driver. Together
//! these run a fine-tune step's forward, backward, AND optimizer
//! entirely on the GPU (paired with [`MlxAdam`](crate::MlxAdam)).

use crate::MlxAdam;
use mlx_rs::Array;
use mlx_rs::error::Result;
use mlx_rs::transforms::value_and_grad_with_argnums;

/// LoRA-adapted linear: `y = x @ (w + scale * (a @ b))`.
///
/// Shapes: `x` is `[n, in]`, `w` is `[in, out]` (frozen base), `a` is
/// `[in, rank]`, `b` is `[rank, out]`; result is `[n, out]`. Building
/// the forward from MLX ops (vs the eager CPU interpreter, which
/// materializes every intermediate) is what lets `value_and_grad`
/// trace a real gradient w.r.t. the adapters `a` and `b`.
pub fn lora_linear(x: &Array, w: &Array, a: &Array, b: &Array, scale: f32) -> Result<Array> {
    let delta = a.matmul(b)?.multiply(Array::from_f32(scale))?;
    x.matmul(w.add(delta)?)
}

/// Evaluate `loss_fn(params)` and its gradient w.r.t. every param,
/// on-device. `loss_fn` returns a single scalar `Array` in a `Vec`
/// (mlx's transform contract). Returns the scalar loss as `f32` plus
/// one gradient `Array` per param (same order).
pub fn loss_and_grads<F>(params: &[Array], loss_fn: F) -> Result<(f32, Vec<Array>)>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let argnums: Vec<i32> = (0..params.len() as i32).collect();
    let mut value_and_grad = value_and_grad_with_argnums(loss_fn, argnums.as_slice());
    let (values, grads) = value_and_grad(params)?;
    Ok((values[0].item::<f32>(), grads))
}

/// Run `steps` Adam steps over `params`, returning the per-step loss
/// curve. Forward + backward (`loss_and_grads`) + the optimizer update
/// (`adam`) all run on-device; nothing round-trips to the CPU.
pub fn train_steps<F>(
    params: &mut [Array],
    adam: &mut MlxAdam,
    steps: usize,
    loss_fn: F,
) -> Result<Vec<f32>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let mut curve = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss, grads) = loss_and_grads(params, &loss_fn)?;
        curve.push(loss);
        adam.step(params, &grads)?;
    }
    Ok(curve)
}

/// Adam hyperparameters + the 1-based step counter `t`. Bundled so
/// callers (and [`adam_update`]) pass one value instead of five.
pub struct AdamHp {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub t: i32,
}

/// One stateless Adam update for a single parameter, returning
/// `(w_new, m_new, v_new)`. For callers that persist moment buffers
/// externally (e.g. the interpreter's per-step MLX LoRA path, which
/// keeps m/v in the eval Environment between `adam(...)` calls) rather
/// than inside an [`MlxAdam`].
pub fn adam_update(
    w: &Array,
    g: &Array,
    m: &Array,
    v: &Array,
    hp: &AdamHp,
) -> Result<(Array, Array, Array)> {
    let m_new = m
        .multiply(Array::from_f32(hp.b1))?
        .add(g.multiply(Array::from_f32(1.0 - hp.b1))?)?;
    let v_new = v
        .multiply(Array::from_f32(hp.b2))?
        .add(g.square()?.multiply(Array::from_f32(1.0 - hp.b2))?)?;
    let mhat = m_new.divide(Array::from_f32(1.0 - hp.b1.powi(hp.t)))?;
    let vhat = v_new.divide(Array::from_f32(1.0 - hp.b2.powi(hp.t)))?;
    let denom = vhat.sqrt()?.add(Array::from_f32(hp.eps))?;
    let w_new = w.subtract(mhat.divide(denom)?.multiply(Array::from_f32(hp.lr))?)?;
    Ok((w_new, m_new, v_new))
}
