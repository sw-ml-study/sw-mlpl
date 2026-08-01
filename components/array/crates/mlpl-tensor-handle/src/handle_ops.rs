//! Resident-op wrappers on [`TensorHandle`]. Policy: an op runs on
//! the device when AT LEAST ONE operand is resident (the host side
//! is uploaded first); two host operands are the CPU path's job
//! and return [`HandleError::NotResident`] -- placement decisions
//! stay with the caller (the tape), never silently here.

use crate::device::{BinKind, Dev, HandleError, UnaryKind};
use crate::handle::TensorHandle;
use crate::registry::require_ops;

impl TensorHandle {
    /// The resident arm, uploading a host value if needed.
    fn as_dev(&self) -> Result<Dev, HandleError> {
        match self {
            Self::Dev(d) => Ok(d.clone()),
            Self::Cpu(a) => require_ops()?.upload(a),
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
        Ok(Self::Dev(require_ops()?.unary(op, d)?))
    }
}
