//! Single-head causal self-attention as a traceable `mlx_rs::Array`
//! forward. (h=1 matches the LoRA demo model `causal_attention(d, 1, _)`;
//! multi-head -- per-head d_k=d/h slabs -- is a later extension.)

use mlx_rs::Array;
use mlx_rs::error::Result;
use mlx_rs::ops::softmax_axis;

/// Additive causal mask `[t, t]`: 0 on and below the diagonal, a large
/// negative value above it, so `softmax` zeroes out future positions.
pub fn causal_mask(t: usize) -> Array {
    let mut m = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = -1e9;
        }
    }
    Array::from_slice(&m, &[t as i32, t as i32])
}

/// Causal self-attention: `softmax((Q Kᵀ)/sqrt(d_k) + mask) V`, then the
/// output projection. `x` is `[T, d]`; `wq`/`wk`/`wv`/`wo` are `[d, d]`;
/// `mask` is `[T, T]` (see [`causal_mask`]). Result is `[T, d]`.
pub fn causal_attention(
    x: &Array,
    wq: &Array,
    wk: &Array,
    wv: &Array,
    wo: &Array,
    mask: &Array,
) -> Result<Array> {
    let (q, k, v) = (x.matmul(wq)?, x.matmul(wk)?, x.matmul(wv)?);
    let scale = Array::from_f32(1.0 / (q.shape()[1] as f32).sqrt());
    let scores = q.matmul(k.transpose()?)?.multiply(scale)?.add(mask)?;
    let attn = softmax_axis(&scores, 1, None)?;
    attn.matmul(v)?.matmul(wo)
}
