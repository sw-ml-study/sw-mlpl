//! MLX (Apple GPU) compute for the device-agnostic GPU optimizer-step seam.
//!
//! The seam (`GpuAdamStep` / `GpuEnv` / `AdamHp` / `DemoLayout` /
//! `LoraNames`) lives in `mlpl-eval`, which runs the interpreter-coupled
//! architecture recognition and then calls a registered `GpuAdamStep`.
//! This crate provides the MLX `GpuAdamStep` impl: it builds `mlx_rs`
//! arrays from the [`mlpl_eval::GpuEnv`] accessor, runs the forward +
//! backward via `value_and_grad` and a stateless MLX Adam, and writes the
//! results back -- no interpreter, no `ModelSpec`. Split out of
//! `mlpl-eval` so it can depend on the seam without a dependency cycle
//! (S4, docs/build-and-workspace-plan.md).
//!
//! Everything is gated on macos/aarch64 + the `mlx` feature; off-target
//! the crate is empty and [`gpu_step`] does not exist.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod mlx_lora;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod mlx_mlp;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod mlx_step;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use mlx_lora::gpu_step;
