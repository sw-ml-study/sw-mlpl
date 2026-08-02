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

/// Structural-op impl (concat + split) for the MLX backend; lives
/// here rather than `ops_impl.rs` to honor that module's function
/// budget. Both hold the process-wide submission lock like every
/// other op.
impl mlpl_tensor_handle::DeviceShapeOps for crate::ops_impl::MlxOps {
    fn concat(
        &self,
        a: &mlpl_tensor_handle::Dev,
        b: &mlpl_tensor_handle::Dev,
        axis: usize,
    ) -> Result<mlpl_tensor_handle::Dev, HandleError> {
        let (aa, bb) = (crate::buf::MlxBuf::of(a)?, crate::buf::MlxBuf::of(b)?);
        let ax = i32::try_from(axis).map_err(backend_err)?;
        let _guard = mlpl_mlx_rt::mlx_op_lock();
        let out = mlx_rs::ops::concatenate_axis(&[aa, bb], ax).map_err(backend_err)?;
        Ok(crate::buf::MlxBuf::wrap(out))
    }

    fn split2(
        &self,
        a: &mlpl_tensor_handle::Dev,
        axis: usize,
        left_size: usize,
    ) -> Result<(mlpl_tensor_handle::Dev, mlpl_tensor_handle::Dev), HandleError> {
        let aa = crate::buf::MlxBuf::of(a)?;
        let ax = i32::try_from(axis).map_err(backend_err)?;
        let cut = i32::try_from(left_size).map_err(backend_err)?;
        let _guard = mlpl_mlx_rt::mlx_op_lock();
        let mut parts = mlx_rs::ops::split_sections(aa, &[cut], ax).map_err(backend_err)?;
        if parts.len() != 2 {
            return Err(HandleError::Backend(format!(
                "split2: expected 2 parts, got {}",
                parts.len()
            )));
        }
        let r = parts.pop().expect("len checked");
        let l = parts.pop().expect("len checked");
        Ok((crate::buf::MlxBuf::wrap(l), crate::buf::MlxBuf::wrap(r)))
    }
}
