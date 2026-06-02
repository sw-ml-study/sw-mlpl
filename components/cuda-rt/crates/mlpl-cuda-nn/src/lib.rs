//! CUDA-backed nn ops -- the candle analog of `mlpl-mlx-rt`'s
//! reductions + normalization. Reductions (`mean`/`reduce_mul`/
//! `argmax`), normalization (`softmax`/`log_softmax`), and the
//! `cross_entropy` loss. (The elementwise activations live in
//! `mlpl-cuda-elementwise` alongside the other unary maps.) Mirrors
//! the MLX/CPU semantics so compiled MLPL swaps runtimes unchanged.
//!
//! Triple-gated like `mlpl-cuda-rt`; off-target exports nothing.
//! This file is a FACADE (CLAUDE.md).

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod loss;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod norm;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod reduce;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use loss::cross_entropy;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use norm::{log_softmax, softmax};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use reduce::{argmax, mean, reduce_mul};
