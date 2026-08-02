//! The value the tape carries: host array or resident handle.

use mlpl_array::DenseArray;

use crate::device::{Dev, HandleError};

/// Either a host f64 array (the bit-exact CPU reference) or a
/// backend-resident array reached through the registered
/// [`crate::DeviceOps`]. Clones of the `Dev` arm are cheap graph
/// references.
#[derive(Debug, Clone)]
pub enum TensorHandle {
    Cpu(DenseArray),
    Dev(Dev),
}

impl TensorHandle {
    /// Host-side shape (lazy-safe on resident handles).
    #[must_use]
    pub fn dims(&self) -> Vec<usize> {
        match self {
            Self::Cpu(a) => a.shape().dims().to_vec(),
            Self::Dev(d) => d.shape().to_vec(),
        }
    }

    /// True when the value lives on a device.
    #[must_use]
    pub fn is_dev(&self) -> bool {
        matches!(self, Self::Dev(_))
    }

    /// Materialize as a host f64 array. On a resident handle this
    /// is THE sync point: it forces the backend's lazy graph.
    #[must_use]
    pub fn to_dense(&self) -> DenseArray {
        match self {
            Self::Cpu(a) => a.clone(),
            Self::Dev(d) => {
                crate::metrics::bump(crate::metrics::SeamEvent::Download);
                d.to_dense()
            }
        }
    }
}

impl TensorHandle {
    /// Resident concat along `axis` (host side auto-uploads when
    /// the other operand is resident).
    ///
    /// # Errors
    /// `NotResident` when both are host; `Backend` on kernel failure.
    pub fn dev_concat(&self, other: &Self, axis: usize) -> Result<Self, HandleError> {
        if !(self.is_dev() || other.is_dev()) {
            return Err(HandleError::NotResident);
        }
        let (a, b) = (self.as_dev()?, other.as_dev()?);
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        Ok(Self::Dev(
            crate::registry::require_ops()?.concat(&a, &b, axis)?,
        ))
    }

    /// Split a resident handle into the two concat halves.
    ///
    /// # Errors
    /// `NotResident` on a host receiver; `Backend` on kernel failure.
    pub fn dev_split2(&self, axis: usize, left_size: usize) -> Result<(Self, Self), HandleError> {
        let Self::Dev(d) = self else {
            return Err(HandleError::NotResident);
        };
        crate::metrics::bump(crate::metrics::SeamEvent::Submit);
        let (l, r) = crate::registry::require_ops()?.split2(d, axis, left_size)?;
        Ok((Self::Dev(l), Self::Dev(r)))
    }
}

impl From<DenseArray> for TensorHandle {
    fn from(a: DenseArray) -> Self {
        Self::Cpu(a)
    }
}
