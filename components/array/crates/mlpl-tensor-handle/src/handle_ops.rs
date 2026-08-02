//! Resident-op wrappers on [`TensorHandle`]. Policy: an op runs on
//! the device when AT LEAST ONE operand is resident (the host side
//! is uploaded first); two host operands are the CPU path's job
//! and return [`HandleError::NotResident`] -- placement decisions
//! stay with the caller (the tape), never silently here.
//! Includes the axis / shape / loss wrappers (merged from the
//! old `handle_axis.rs` to honor the crate module budget).

use crate::device::{AxisKind, BinKind, Dev, HandleError, UnaryKind};
use crate::handle::TensorHandle;
use crate::registry::require_ops;

impl TensorHandle {
    /// The resident arm, uploading a host value if needed.
    pub(crate) fn as_dev(&self) -> Result<Dev, HandleError> {
        match self {
            Self::Dev(d) => Ok(d.clone()),
            Self::Cpu(a) => {
                let dev = require_ops()?.upload(a)?;
                crate::metrics::bump(crate::metrics::SeamEvent::Upload);
                Ok(dev)
            }
        }
    }

    /// Binary op on the device (see module policy).
    ///
    /// # Errors
    /// `NotResident` when both operands are host-side; backend
    /// errors pass through.
    pub fn dev_binary(&self, op: BinKind, other: &Self) -> Result<Self, HandleError> {
        if !self.is_dev() && !other.is_dev() {
            return Err(HandleError::NotResident);
        }
        let out = require_ops()?.binary(op, &self.as_dev()?, &other.as_dev()?)?;
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(out))
    }

    /// Unary op on a resident handle.
    ///
    /// # Errors
    /// `NotResident` on a host-side receiver.
    pub fn dev_unary(&self, op: UnaryKind) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        let out = require_ops()?.unary(op, d)?;
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(out))
    }

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
        let out = require_ops()?.axis_op(op, d, axis, keep_dims)?;
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(out))
    }

    /// Reshape a resident handle.
    ///
    /// # Errors
    /// `NotResident` on a host-side receiver.
    pub fn dev_reshape(&self, dims: &[usize]) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        let out = require_ops()?.reshape(d, dims)?;
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(out))
    }

    /// Fused cross-entropy forward on resident logits.
    ///
    /// # Errors
    /// `NotResident` on host-side logits.
    pub fn dev_cross_entropy(&self, targets: &[usize]) -> Result<Self, HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        let out = require_ops()?.cross_entropy(d, targets)?;
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(out))
    }
}
