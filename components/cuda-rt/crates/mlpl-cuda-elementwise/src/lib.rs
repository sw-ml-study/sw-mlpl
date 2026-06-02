//! CUDA-backed elementwise primitives -- the candle analog of
//! `mlpl-mlx-rt::elementwise`. Binary ops (`add`/`sub`/`mul`/`div`)
//! with scalar broadcasting plus unary `neg`, over candle Tensors,
//! reusing `mlpl-cuda-rt`'s device + conversion plumbing.
//!
//! Triple-gated like `mlpl-cuda-rt`; off-target the crate exports
//! nothing. This file is a FACADE (CLAUDE.md).

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod arith;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod dispatch;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod unary;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use arith::{add, div, mul, sub};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use unary::neg;
