//! The resident MLX array behind a `DeviceArray` handle.

use std::any::Any;
use std::sync::Arc;

use mlpl_array::{DenseArray, Shape};
use mlpl_tensor_handle::{Dev, DeviceArray, HandleError};
use mlx_rs::Array as MlxArray;

/// An MLX array (possibly an UNEVALUATED lazy graph node) plus its
/// host-side dims. Shape is known without forcing the graph -- MLX
/// does shape inference eagerly.
#[derive(Debug)]
pub struct MlxBuf {
    arr: MlxArray,
    dims: Vec<usize>,
}

// SAFETY: mlx_rs::Array is Send but not Sync (the underlying C++
// object is not internally synchronized). MLPL's discipline is that
// EVERY MLX submission, graph construction, and eval in this
// process holds mlpl_mlx_rt::mlx_op_lock() -- enforced by every op
// in mlpl-mlx-rt, every DeviceOps method in this crate, and
// MlxBuf::to_dense below -- so cross-thread access is always
// serialized.
unsafe impl Sync for MlxBuf {}

impl MlxBuf {
    /// Wrap a (lazy) MLX array as a shared handle. Dims come from
    /// MLX's eager shape inference, so no eval happens here.
    pub(crate) fn wrap(arr: MlxArray) -> Dev {
        let dims: Vec<usize> = arr
            .shape()
            .iter()
            .map(|&d| usize::try_from(d).expect("mlx dims are non-negative"))
            .collect();
        Arc::new(Self { arr, dims })
    }

    /// Borrow the underlying MLX array from a generic handle.
    ///
    /// # Errors
    /// [`HandleError::Backend`] when the handle belongs to another
    /// backend.
    pub(crate) fn of(dev: &Dev) -> Result<&MlxArray, HandleError> {
        dev.as_any()
            .downcast_ref::<Self>()
            .map(|b| &b.arr)
            .ok_or_else(|| HandleError::Backend("handle is not an MLX buffer".into()))
    }
}

impl DeviceArray for MlxBuf {
    fn shape(&self) -> &[usize] {
        &self.dims
    }

    /// THE sync point: forces the lazy graph (under the process
    /// lock) and widens f32 -> f64. The download goes through a
    /// flatten-reshape so VIEW tips (transpose etc.) materialize in
    /// LOGICAL row-major order -- `as_slice` on a bare view would
    /// read the underlying buffer in physical order (the same
    /// hazard mlpl-mlx-rt's shapes.rs documents). For contiguous
    /// graphs the reshape is a free no-op node.
    fn to_dense(&self) -> DenseArray {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let flat = self
            .arr
            .reshape(&[-1])
            .expect("mlx flatten-reshape cannot fail on a shape-inferred graph");
        flat.eval()
            .expect("mlx eval on shape-inferred graph should not fail");
        let out: &[f32] = flat.as_slice();
        let data: Vec<f64> = out.iter().map(|&x| f64::from(x)).collect();
        DenseArray::new(Shape::new(self.dims.clone()), data)
            .expect("dims tracked from MLX shape inference always match")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
