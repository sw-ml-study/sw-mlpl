//! The assembled demo forward + its frozen base weights.

use mlpl_mlx_forward::{causal_attention, cross_entropy, embed, rms_norm};
use mlpl_mlx_train::lora_linear;
use mlx_rs::Array;
use mlx_rs::error::Result;

/// Frozen base weights for the demo model. The 4 attention projections
/// and the head are LoRA-wrapped; their adapters are passed separately
/// to [`demo_forward`] as the traced params.
pub struct DemoWeights {
    pub embed: Array,  // [V, d]
    pub gamma1: Array, // [d] (pre-attention RMSNorm)
    pub wq: Array,     // [d, d]
    pub wk: Array,     // [d, d]
    pub wv: Array,     // [d, d]
    pub wo: Array,     // [d, d]
    pub gamma2: Array, // [d] (final RMSNorm)
    pub head: Array,   // [d, V]
    pub mask: Array,   // [T, T] causal mask
    pub scale: f32,    // LoRA alpha/rank
    pub eps: f32,      // RMSNorm epsilon
}

/// Effective LoRA weight `w + scale * (a @ b)` -- traced w.r.t. `a`,`b`.
fn lora_merge(w: &Array, a: &Array, b: &Array, scale: f32) -> Result<Array> {
    w.add(a.matmul(b)?.multiply(Array::from_f32(scale))?)
}

/// Run the demo forward on one-hot inputs `[T, V]` and return the mean
/// cross-entropy against one-hot targets `[T, V]`. `adapters` is 10
/// Arrays: A/B for wq, wk, wv, wo, head (in that order). The whole
/// computation is one MLX graph differentiable w.r.t. the adapters.
pub fn demo_forward(
    w: &DemoWeights,
    adapters: &[Array],
    x_onehot: &Array,
    y_onehot: &Array,
) -> Result<Array> {
    let s = w.scale;
    let m = |i: usize, base: &Array| lora_merge(base, &adapters[i], &adapters[i + 1], s);
    let emb = embed(x_onehot, &w.embed)?;
    let n1 = rms_norm(&emb, &w.gamma1, w.eps)?;
    let attn = causal_attention(
        &n1,
        &m(0, &w.wq)?,
        &m(2, &w.wk)?,
        &m(4, &w.wv)?,
        &m(6, &w.wo)?,
        &w.mask,
    )?;
    let n2 = rms_norm(&emb.add(attn)?, &w.gamma2, w.eps)?;
    let logits = lora_linear(&n2, &w.head, &adapters[8], &adapters[9], s)?;
    cross_entropy(&logits, y_onehot)
}
