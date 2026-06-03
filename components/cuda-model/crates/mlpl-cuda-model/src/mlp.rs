//! The tic-tac-toe board-policy forward: a two-layer MLP whose BOTH
//! linears are LoRA-adapted, assembled into one traceable candle graph
//! so a fine-tune step runs forward + backward + optimizer on the GPU.
//! The CUDA analog of `mlpl-mlx-model::mlp`.
//!
//! Matches the CPU `apply_model` for `Chain[LinearLora, relu,
//! LinearLora]`: `h = relu(lora(x, W1) + b1)`, `logits = lora(h, W2) +
//! b2`, mean softmax cross-entropy. The frozen bases (W1, b1, W2, b2)
//! live in [`MlpWeights`]; the four adapters (A1, B1, A2, B2) are the
//! traced params.

use candle_core::{Result, Tensor};
use mlpl_cuda_forward::{cross_entropy, lora_linear};

/// Frozen base weights for the board-policy MLP. The trained params are
/// the two layers' `LoRA` adapters, passed to [`mlp_forward`] separately.
pub struct MlpWeights {
    /// Layer-1 frozen base `[in, hidden]`.
    pub w1: Tensor,
    /// Layer-1 bias `[1, hidden]` (broadcast).
    pub b1: Tensor,
    /// Head frozen base `[hidden, classes]`.
    pub w2: Tensor,
    /// Head bias `[1, classes]` (broadcast).
    pub b2: Tensor,
    /// Layer-1 `LoRA` `alpha / rank`.
    pub scale1: f64,
    /// Head `LoRA` `alpha / rank`.
    pub scale2: f64,
}

/// Forward on board features `x` `[N, in]` and one-hot move labels
/// `y_onehot` `[N, classes]`; returns the mean cross-entropy as a scalar.
/// `adapters` is `[A1, B1, A2, B2]` (layer 1 then head) -- the only
/// traced params. The whole graph is differentiable w.r.t. them.
///
/// # Errors
/// Propagates candle errors from any forward primitive.
pub fn mlp_forward(
    w: &MlpWeights,
    adapters: &[Tensor],
    x: &Tensor,
    y_onehot: &Tensor,
) -> Result<Tensor> {
    let z1 = lora_linear(x, &w.w1, &adapters[0], &adapters[1], w.scale1)?.broadcast_add(&w.b1)?;
    let h = z1.relu()?;
    let logits =
        lora_linear(&h, &w.w2, &adapters[2], &adapters[3], w.scale2)?.broadcast_add(&w.b2)?;
    cross_entropy(&logits, y_onehot)
}
