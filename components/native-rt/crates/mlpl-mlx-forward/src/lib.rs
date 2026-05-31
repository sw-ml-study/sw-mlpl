//! MLX-native forward primitives for the LoRA fine-tune model.
//!
//! Each function builds part of the model forward from `mlx_rs::Array`
//! ops so the whole forward+loss is ONE lazy MLX graph that
//! `value_and_grad` can differentiate (the MLPL interpreter forward is
//! eager and CPU-materialized, so it cannot be traced as-is). Paired
//! with `mlpl-mlx-train`'s `loss_and_grads` + `MlxAdam` to run a
//! fine-tune step entirely on the GPU.
//!
//! Gated like the other MLX crates (the `mlx` feature, macOS,
//! aarch64); off-target the crate builds but exports nothing.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod forward;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use attention::{causal_attention, causal_mask};
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use forward::{cross_entropy, embed, rms_norm};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod forward_tests;
