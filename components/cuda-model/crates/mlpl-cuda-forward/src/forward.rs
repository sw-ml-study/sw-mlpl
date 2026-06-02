//! Non-attention traceable primitives: embed, `RMSNorm`, cross-entropy.

use candle_core::{Result, Tensor};

/// Embedding lookup as a one-hot matmul: `onehot [T, V] @ table [V, d]`
/// -> `[T, d]`. The one-hot form (vs an index gather) keeps it
/// differentiable w.r.t. `table`.
///
/// # Errors
/// Propagates candle shape/matmul errors.
pub fn embed(onehot: &Tensor, table: &Tensor) -> Result<Tensor> {
    onehot.matmul(table)
}

/// `RMSNorm` over the last axis (gamma-free in the demo: `gamma` is the
/// all-ones `[d]` vector): `x / sqrt(mean(x^2) + eps) * gamma`. `x` is
/// `[T, d]`, `gamma` is `[d]` (broadcast); result is `[T, d]`. `eps`
/// matches the CPU path's 1e-8.
///
/// # Errors
/// Propagates candle errors.
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    let inv = x
        .sqr()?
        .mean_keepdim(1)?
        .affine(1.0, f64::from(eps))?
        .sqrt()?;
    x.broadcast_div(&inv)?.broadcast_mul(gamma)
}

/// Mean per-row softmax cross-entropy. `logits` is `[T, V]`,
/// `targets_onehot` is `[T, V]` (one-hot rows). Numerically stable via
/// `logsumexp`; returns a scalar. Fully traceable (no gather), so it is
/// differentiable w.r.t. `logits`.
///
/// # Errors
/// Propagates candle errors.
pub fn cross_entropy(logits: &Tensor, targets_onehot: &Tensor) -> Result<Tensor> {
    let m = logits.max_keepdim(1)?;
    let lse = logits
        .broadcast_sub(&m)?
        .exp()?
        .sum_keepdim(1)?
        .log()?
        .broadcast_add(&m)?;
    let picked = logits.mul(targets_onehot)?.sum_keepdim(1)?;
    lse.sub(&picked)?.mean_all()
}
