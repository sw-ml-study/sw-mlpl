//! Single-head causal self-attention as a traceable candle forward
//! (h=1 matches the `LoRA` demo's `causal_attention(d, 1, _)`).

use candle_core::{Device, Result, Tensor};

/// Additive causal mask `[t, t]`: 0 on/below the diagonal, a large
/// negative value above it, so `softmax` zeroes future positions.
///
/// # Errors
/// Propagates candle tensor-construction errors.
pub fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut m = vec![0.0f32; t * t];
    for (i, row) in m.chunks_mut(t).enumerate() {
        for cell in row.iter_mut().skip(i + 1) {
            *cell = -1e9;
        }
    }
    Tensor::from_vec(m, (t, t), device)
}

/// Causal self-attention: `softmax((Q Kᵀ)/sqrt(d_k) + mask) V`, then the
/// output projection. `x` is `[T, d]`; `wq`/`wk`/`wv`/`wo` are `[d, d]`;
/// `mask` is `[T, T]`. Result is `[T, d]`.
///
/// # Errors
/// Propagates candle errors.
#[allow(clippy::cast_precision_loss)] // d_k is a small head dimension; exact in f64.
pub fn causal_attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let (q, k, v) = (x.matmul(wq)?, x.matmul(wk)?, x.matmul(wv)?);
    let d_k = q.dim(1)?;
    let scale = 1.0 / (d_k as f64).sqrt();
    let scores = q.matmul(&k.t()?)?.affine(scale, 0.0)?.broadcast_add(mask)?;
    let attn = candle_nn::ops::softmax(&scores, 1)?;
    attn.matmul(&v)?.matmul(wo)
}
