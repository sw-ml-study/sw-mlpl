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
mod mlp;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod model;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use mlp::{MlpWeights, mlp_forward};
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use model::{DemoWeights, demo_forward};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod mlp_tests;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod model_tests;

// Metal submissions from parallel test threads SIGSEGV (saga E4
// step 001, mlx-rs 0.25.3) -- same hazard mlpl-mlx-train already
// guards. Every test serializes on this lock (poison-tolerant so
// one failing test cannot wedge the rest).
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
pub(crate) static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
