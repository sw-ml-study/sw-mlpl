//! The assembled demo forward + its frozen base weights.
//!
//! Matches the CPU `apply_model` for the demo architecture exactly so
//! the MLX path is parity-comparable: RMSNorm is gamma-free with
//! eps=1e-8 (the `RmsNorm` layer is parameter-free); the attention
//! projections live in the `Attention` variant (NOT `Linear`), so
//! `lora()` leaves them frozen -- only the head `Linear` becomes
//! `LinearLora`. Hence a single trained adapter pair (A, B) and a head
//! bias, not one pair per projection.

use mlpl_mlx_forward::{causal_attention, cross_entropy, embed, rms_norm};
use mlpl_mlx_train::lora_linear;
use mlx_rs::Array;
use mlx_rs::error::Result;

/// Frozen base weights for the demo model. Everything here is frozen;
/// the only trained params are the head LoRA adapters, passed to
/// [`demo_forward`] separately. `gamma` is an all-ones `[d]` vector
/// (RMSNorm is gamma-free -- ones make the scale a no-op while reusing
/// the `rms_norm` primitive).
pub struct DemoWeights {
    pub embed: Array,  // [V, d]
    pub wq: Array,     // [d, d]
    pub wk: Array,     // [d, d]
    pub wv: Array,     // [d, d]
    pub wo: Array,     // [d, d]
    pub head_w: Array, // [d, V] frozen base of the head Linear
    pub head_b: Array, // [1, V] head bias (broadcast)
    pub gamma: Array,  // [d] ones
    pub mask: Array,   // [T, T] causal mask
    pub scale: f32,    // head LoRA alpha / rank
    pub eps: f32,      // RMSNorm epsilon (1e-8, matching CPU)
}

/// Run the demo forward on one-hot inputs `[T, V]` and return the mean
/// cross-entropy against one-hot targets `[T, V]`. `adapters` is the
/// head's `[A, B]` pair -- the only traced params; everything in `w` is
/// frozen. The whole computation is one MLX graph differentiable w.r.t.
/// `adapters`.
pub fn demo_forward(
    w: &DemoWeights,
    adapters: &[Array],
    x_onehot: &Array,
    y_onehot: &Array,
) -> Result<Array> {
    let emb = embed(x_onehot, &w.embed)?;
    let n1 = rms_norm(&emb, &w.gamma, w.eps)?;
    let attn = causal_attention(&n1, &w.wq, &w.wk, &w.wv, &w.wo, &w.mask)?;
    let n2 = rms_norm(&emb.add(attn)?, &w.gamma, w.eps)?;
    let logits =
        lora_linear(&n2, &w.head_w, &adapters[0], &adapters[1], w.scale)?.add(&w.head_b)?;
    cross_entropy(&logits, y_onehot)
}
