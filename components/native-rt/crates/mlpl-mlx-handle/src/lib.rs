//! MLX backend for the device-resident tensor seam (saga E4 step
//! 002). See `mlpl-tensor-handle` for the contract. Everything is
//! triple-gated: on non-Apple targets this crate exports nothing.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod buf;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod ops_impl;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod support;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use buf::MlxBuf;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use support::register_mlx_device_ops;
