//! Registration + error plumbing for the MLX backend.

use std::sync::Arc;

use mlpl_tensor_handle::HandleError;

/// Map any mlx-rs error into the seam's error type.
pub(crate) fn backend_err<E: std::fmt::Display>(e: E) -> HandleError {
    HandleError::Backend(e.to_string())
}

/// Register MLX as the process's device backend (idempotent).
pub fn register_mlx_device_ops() {
    mlpl_tensor_handle::register_device_ops(Arc::new(crate::ops_impl::MlxOps));
}

/// `1.0` where `x > 0`, else `0.0` (relu backward mask), in f32.
pub(crate) fn step_mask(x: &mlx_rs::Array) -> Result<mlx_rs::Array, mlx_rs::error::Exception> {
    x.gt(mlx_rs::Array::from_f32(0.0))?.as_type::<f32>()
}
