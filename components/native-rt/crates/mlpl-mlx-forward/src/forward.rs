//! Traceable forward primitives in `mlx_rs::Array` ops.

use mlx_rs::Array;
use mlx_rs::error::Result;

/// Embedding lookup as a one-hot matmul: `onehot [T, V] @ table [V, d]`
/// -> `[T, d]`. The one-hot form (vs an index gather) keeps the op
/// differentiable w.r.t. `table` and traceable in the MLX graph.
pub fn embed(onehot: &Array, table: &Array) -> Result<Array> {
    onehot.matmul(table)
}

/// RMSNorm over the last axis: `x / sqrt(mean(x^2) + eps) * gamma`.
/// `x` is `[T, d]`, `gamma` is `[d]` (broadcast); result is `[T, d]`.
pub fn rms_norm(x: &Array, gamma: &Array, eps: f32) -> Result<Array> {
    let ms = x.square()?.mean_axis(1, true)?;
    let inv = ms.add(Array::from_f32(eps))?.sqrt()?;
    x.divide(inv)?.multiply(gamma)
}

/// Mean per-row softmax cross-entropy. `logits` is `[T, V]`,
/// `targets_onehot` is `[T, V]` (one-hot rows). Uses the numerically
/// stable `logsumexp`; returns a scalar `Array`.
pub fn cross_entropy(logits: &Array, targets_onehot: &Array) -> Result<Array> {
    let lse = logits.logsumexp_axis(1, true)?;
    let picked = logits.multiply(targets_onehot)?.sum_axis(1, true)?;
    lse.subtract(picked)?.mean(None)
}
