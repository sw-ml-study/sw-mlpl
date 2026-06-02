//! Traceable LoRA-adapted linear (the CUDA analog of
//! `mlpl-mlx-train::lora_linear`).

use candle_core::{Result, Tensor};

/// `LoRA` linear `x @ (w + scale * (a @ b))`. `x` is `[T, in]`, `w` is
/// `[in, out]`, `a` is `[in, rank]`, `b` is `[rank, out]`. Folding the
/// adapter delta into `w` matches `x @ w + scale * (x @ a @ b)` while
/// keeping one matmul. Differentiable w.r.t. `a` and `b`.
///
/// # Errors
/// Propagates candle shape/matmul errors.
pub fn lora_linear(x: &Tensor, w: &Tensor, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
    let delta = a.matmul(b)?.affine(scale, 0.0)?;
    x.matmul(&w.add(&delta)?)
}
