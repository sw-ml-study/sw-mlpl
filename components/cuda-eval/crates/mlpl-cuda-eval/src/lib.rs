//! CUDA compute for the device-agnostic GPU optimizer-step seam.
//!
//! The seam (`GpuAdamStep` / `GpuEnv` / `AdamHp` / `DemoLayout` /
//! `LoraNames`) lives in `mlpl-eval`, which runs the interpreter-coupled
//! architecture recognition and then calls a registered `GpuAdamStep`.
//! This crate provides the CUDA `GpuAdamStep` impl: it builds candle
//! tensors from the [`mlpl_eval::GpuEnv`] accessor, runs the forward +
//! backward via candle autograd and a stateless candle Adam, and writes
//! the results back -- no interpreter, no `ModelSpec`. Split out of
//! `mlpl-eval` so it can depend on the seam without a dependency cycle
//! (S3, docs/build-and-workspace-plan.md).
//!
//! Everything is gated on linux/x86_64 + the `cuda` feature; off-target
//! the crate is empty and [`gpu_step`] does not exist.

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda_lora;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda_mlp;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda_step;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub use cuda_lora::gpu_step;
