//! The tic-tac-toe board-policy forward: a two-layer MLP whose BOTH
//! linears are LoRA-adapted, assembled into one traceable MLX graph so
//! a fine-tune step runs forward + backward + optimizer on the GPU.
//!
//! Matches the CPU `apply_model` for `Chain[LinearLora, relu,
//! LinearLora]`: `h = relu(lora(x, W1) + b1)`, `logits = lora(h, W2) +
//! b2`, mean softmax cross-entropy against one-hot move labels. The
//! frozen bases (W1, b1, W2, b2) live in [`MlpWeights`]; the four
//! adapters (A1, B1, A2, B2) are the traced params.

use mlpl_mlx_forward::cross_entropy;
use mlpl_mlx_train::lora_linear;
use mlx_rs::Array;
use mlx_rs::error::Result;

/// Frozen base weights for the board-policy MLP. The trained params are
/// the two layers' LoRA adapters, passed to [`mlp_forward`] separately.
pub struct MlpWeights {
    pub w1: Array,   // [in, hidden] frozen base of layer 1
    pub b1: Array,   // [1, hidden] bias (broadcast)
    pub w2: Array,   // [hidden, classes] frozen base of the head
    pub b2: Array,   // [1, classes] head bias
    pub scale1: f32, // layer-1 LoRA alpha / rank
    pub scale2: f32, // head LoRA alpha / rank
}

/// Forward on board features `x` `[N, in]` and one-hot move labels
/// `y_onehot` `[N, classes]`; returns the mean cross-entropy as a scalar
/// `Array`. `adapters` is `[A1, B1, A2, B2]` (layer 1 then head) -- the
/// only traced params. The whole graph is differentiable w.r.t. them.
pub fn mlp_forward(
    w: &MlpWeights,
    adapters: &[Array],
    x: &Array,
    y_onehot: &Array,
) -> Result<Array> {
    let z1 = lora_linear(x, &w.w1, &adapters[0], &adapters[1], w.scale1)?.add(&w.b1)?;
    let h = mlx_rs::nn::relu(&z1)?;
    let logits = lora_linear(&h, &w.w2, &adapters[2], &adapters[3], w.scale2)?.add(&w.b2)?;
    cross_entropy(&logits, y_onehot)
}
