//! CUDA-backed runtime target for MLPL -- the candle analog of
//! `mlpl-mlx-rt`. This crate holds the conversion plumbing, matmul,
//! and shape ops; elementwise arithmetic lives in the sibling
//! `mlpl-cuda-elementwise`, and the nn surface (activations,
//! reductions, softmax, `cross_entropy`) in `mlpl-cuda-nn`.
//!
//! A program compiled against `mlpl-rt::<op>` is swappable for
//! `mlpl_cuda_rt::<op>`; outputs agree within a documented fp32
//! tolerance (candle computes in f32 on the GPU, the CPU path f64).
//!
//! CUDA code is triple-gated: the `cuda` Cargo feature, plus
//! `target_os = "linux"` and `target_arch = "x86_64"`. On any host
//! that fails a gate the crate still builds but exports no
//! primitives -- the CPU path in `mlpl-rt` remains authoritative.
//!
//! This file is a FACADE (CLAUDE.md): behavior lives in named files.

pub use mlpl_array::{ArrayError, DenseArray, Shape};

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod convert;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod linalg;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod shape;

// Conversion helpers + the candle device, re-exported so sibling
// crates (mlpl-cuda-elementwise, mlpl-cuda-nn, mlpl-eval's CUDA
// path) share one device and the f64 <-> f32 boundary.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use convert::{Labels, cuda_device, cuda_to_dense_data, dense_to_cuda, finalize};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use linalg::matmul;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use shape::{reshape, transpose};
