//! `mlpl-cuda-train` -- CUDA on-device training primitives (the
//! candle "approach A" analog of `mlpl-mlx-train`).
//!
//! Step 001 is a GO/NO-GO spike: prove candle's CUDA backend builds
//! and runs autodiff + an optimizer on this host's GPU (RTX 5060 Ti,
//! Blackwell `sm_120`, CUDA 13.2). Everything is gated behind the
//! `cuda` feature AND `target_os = "linux"`; off-target the crate is
//! empty so the workspace stays cross-platform-buildable.
//!
//! This file is a FACADE (CLAUDE.md): behavior lives in `spike.rs`.

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod spike;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use spike::{cuda_device, grad_at_zero, train_adam};
