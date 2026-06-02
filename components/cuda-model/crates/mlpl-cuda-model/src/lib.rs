//! The assembled demo `LoRA` fine-tune model forward -- the CUDA analog
//! of `mlpl-mlx-model`. One traceable candle graph matching the CPU
//! `apply_model` for the demo architecture: gamma-free `RMSNorm`
//! (eps=1e-8), attention projections frozen (only the head is
//! LoRA-adapted), head carries a bias. The frozen base weights live in
//! [`DemoWeights`]; the head `[A, B]` adapter pair is the only traced
//! param, so candle autograd differentiates the whole graph w.r.t. it.
//!
//! Triple-gated (the `cuda` feature + Linux + `x86_64`); off-target the
//! crate exports nothing. This file is a FACADE (CLAUDE.md).

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod model;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use model::{DemoWeights, demo_forward};
