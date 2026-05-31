//! The demo LoRA fine-tune model forward, assembled from MLX
//! primitives into ONE traceable graph.
//!
//! Mirrors the demo architecture
//! `embed -> residual(rms_norm -> causal_attention) -> rms_norm ->
//! lora_linear head -> cross_entropy`. The frozen base weights are
//! captured in [`DemoWeights`]; the LoRA adapters (one A/B pair per
//! linear -- the 4 attention projections + the head) are the traced
//! params. Paired with `mlpl-mlx-train`'s `loss_and_grads` + `MlxAdam`,
//! a fine-tune step runs forward + backward + optimizer entirely on the
//! GPU.
//!
//! Gated like the other MLX crates (the `mlx` feature, macOS, aarch64);
//! off-target the crate builds but exports nothing.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod model;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use model::{DemoWeights, demo_forward};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod model_tests;
