//! The value the tape carries: host array or resident handle.

use mlpl_array::DenseArray;

use crate::device::Dev;

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

impl From<DenseArray> for TensorHandle {
    fn from(a: DenseArray) -> Self {
        Self::Cpu(a)
    }
}
