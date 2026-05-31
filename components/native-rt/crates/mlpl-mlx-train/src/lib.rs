//! MLX on-device training primitives (approach A).
//!
//! The MLX fine-tune loop runs FULLY on the GPU by expressing the
//! per-step loss as an `mlx_rs`-traceable closure and differentiating
//! it with `mlx_rs::transforms::value_and_grad` -- no hand-written
//! backward formulas, no CPU round-trip. The companion [`MlxAdam`]
//! keeps its first/second moment buffers as `mlx_rs::Array` and does
//! the parameter update with MLX elementwise ops, so gradients and
//! optimizer state never leave the device. [`train_steps`] ties them
//! into a step loop; [`lora_linear`] is the traceable LoRA forward.
//!
//! Sibling to the forward-only `mlpl-mlx-rt`. Gated behind the same
//! three conditions (the `mlx` feature, `target_os = "macos"`,
//! `target_arch = "aarch64"`); on any other host the crate builds but
//! exports nothing, keeping cross-platform CI green.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod adam;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod kernel;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use adam::MlxAdam;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use kernel::{AdamHp, adam_update, lora_linear, loss_and_grads, train_steps};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod kernel_tests;

// MLX's `value_and_grad` transform mutates a global trace/graph state
// and is NOT safe to run concurrently. cargo runs `#[test]`s in
// parallel, so every test that builds a gradient must serialize on
// this lock (poison-tolerant: a failing test must not wedge the rest).
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
pub(crate) static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
