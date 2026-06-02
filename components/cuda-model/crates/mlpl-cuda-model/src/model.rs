//! The assembled demo forward + its frozen base weights.
//!
//! Matches the CPU `apply_model` for the demo architecture exactly:
//! `RMSNorm` is gamma-free with eps=1e-8 (`gamma` is an all-ones `[d]`
//! vector so the scale is a no-op while reusing `rms_norm`); the
//! attention projections are frozen (they live in the `Attention`
//! variant, not `Linear`, so `lora()` leaves them alone); only the head
//! `Linear` becomes `LinearLora`, carrying one `[A, B]` adapter pair and
//! a bias.

use candle_core::{Result, Tensor};
use mlpl_cuda_forward::{causal_attention, cross_entropy, embed, lora_linear, rms_norm};

/// Frozen base weights for the demo model. The only trained params are
/// the head `LoRA` adapters, passed to [`demo_forward`] separately.
/// `gamma` is an all-ones `[d]` vector (gamma-free `RMSNorm`).
pub struct DemoWeights {
    /// Embedding table `[V, d]`.
    pub embed: Tensor,
    /// Attention query projection `[d, d]`.
    pub wq: Tensor,
    /// Attention key projection `[d, d]`.
    pub wk: Tensor,
    /// Attention value projection `[d, d]`.
    pub wv: Tensor,
    /// Attention output projection `[d, d]`.
    pub wo: Tensor,
    /// Frozen base of the head `Linear` `[d, V]`.
    pub head_w: Tensor,
    /// Head bias `[1, V]` (broadcast).
    pub head_b: Tensor,
    /// All-ones `[d]` `RMSNorm` scale.
    pub gamma: Tensor,
    /// Causal mask `[T, T]`.
    pub mask: Tensor,
    /// Head `LoRA` `alpha / rank`.
    pub scale: f64,
    /// `RMSNorm` epsilon (1e-8, matching CPU).
    pub eps: f32,
}

/// Run the demo forward on one-hot inputs `[T, V]` and return the mean
/// cross-entropy against one-hot targets `[T, V]`. `adapters` is the
/// head's `[A, B]` pair -- the only traced params; everything in `w` is
/// frozen. The whole computation is one candle graph differentiable
/// w.r.t. `adapters`.
///
/// # Errors
/// Propagates candle errors from any forward primitive.
pub fn demo_forward(
    w: &DemoWeights,
    adapters: &[Tensor],
    x_onehot: &Tensor,
    y_onehot: &Tensor,
) -> Result<Tensor> {
    let emb = embed(x_onehot, &w.embed)?;
    let n1 = rms_norm(&emb, &w.gamma, w.eps)?;
    let attn = causal_attention(&n1, &w.wq, &w.wk, &w.wv, &w.wo, &w.mask)?;
    let n2 = rms_norm(&emb.add(&attn)?, &w.gamma, w.eps)?;
    let logits = lora_linear(&n2, &w.head_w, &adapters[0], &adapters[1], w.scale)?
        .broadcast_add(&w.head_b)?;
    cross_entropy(&logits, y_onehot)
}
