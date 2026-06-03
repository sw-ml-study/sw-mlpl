//! `mlpl-cuda-train` -- CUDA on-device training kernel (the candle
//! "approach A" analog of `mlpl-mlx-train`): candle autodiff
//! (`loss_and_grads`) + a stateless Adam update (`adam_update`) whose
//! moment buffers are candle `Tensor`s. Paired with the interpreter's
//! per-step CUDA `LoRA` path, a fine-tune step runs forward + backward +
//! optimizer on the GPU.
//!
//! Everything is gated behind the `cuda` feature AND `target_os =
//! "linux"` + `target_arch = "x86_64"`; off-target the crate is empty
//! so the workspace stays cross-platform-buildable.
//!
//! This file is a FACADE (CLAUDE.md): behavior lives in `kernel.rs`.

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod kernel;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use kernel::{AdamHp, adam_update, loss_and_grads};
