//! Traceable forward primitives in candle `Tensor` ops -- the CUDA
//! analog of `mlpl-mlx-forward` (plus the `lora_linear` that lived in
//! `mlpl-mlx-train`). Every op is differentiable, so candle autograd
//! flows through the assembled demo model and a fine-tune step trains
//! on the GPU. Formulas match the CPU `apply_model` exactly (gamma-free
//! `RMSNorm` eps=1e-8, single-head causal attention scale=1/sqrt(`d_k`),
//! mean per-row softmax cross-entropy against one-hot targets).
//!
//! Triple-gated (the `cuda` feature + Linux + `x86_64`); off-target
//! the crate exports nothing. This file is a FACADE (CLAUDE.md).

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod attention;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod forward;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod lora;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use attention::{causal_attention, causal_mask};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use forward::{cross_entropy, embed, rms_norm};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use lora::lora_linear;
