//! Axis / shape / loss wrappers on [`TensorHandle`] -- the second
//! half of the resident-op surface (split from `handle_ops.rs` to
//! honor the module function budget).

use crate::device::{AxisKind, HandleError};
use crate::handle::TensorHandle;
use crate::registry::require_ops;

impl TensorHandle {
    /// Axis op (softmax / reductions) on a resident handle.
    ///
    /// # Errors
    /// `NotResident` on a host-side receiver.
    pub fn dev_axis(
        &self,
        op: AxisKind,
        axis: Option<usize>,
        keep_dims: bool,
    ) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        Ok(Self::Dev(require_ops()?.axis_op(op, d, axis, keep_dims)?))
    }

    /// Reshape a resident handle.
    ///
    /// # Errors
    /// `NotResident` on a host-side receiver.
    pub fn dev_reshape(&self, dims: &[usize]) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        Ok(Self::Dev(require_ops()?.reshape(d, dims)?))
    }

    /// Fused cross-entropy forward on resident logits.
    ///
    /// # Errors
    /// `NotResident` on host-side logits.
    pub fn dev_cross_entropy(&self, targets: &[usize]) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        Ok(Self::Dev(require_ops()?.cross_entropy(d, targets)?))
    }
}
