//! The backend op contract: the surface the autograd tape needs
//! on both the forward and the backward pass, one method per op
//! family (kind enums keep it narrow).

use mlpl_array::DenseArray;

use crate::device::{AxisKind, BinKind, Dev, HandleError, UnaryKind};

/// The registered backend: uploads host arrays and runs the tape
/// op surface on resident handles. Implementations may be lazy --
/// nothing here promises materialization except
/// [`DeviceArray::to_dense`] on a result.
///
/// # Errors
/// Every method returns [`HandleError::Backend`] when the backend
/// rejects the op (shape mismatch, foreign handle, kernel error).
pub trait DeviceOps: Send + Sync {
    /// Move a host array onto the device (f64 -> backend dtype).
    ///
    /// # Errors
    /// [`HandleError::Backend`] on upload failure.
    fn upload(&self, a: &DenseArray) -> Result<Dev, HandleError>;
    /// Elementwise/matmul binary on two resident arrays.
    ///
    /// # Errors
    /// [`HandleError::Backend`] on shape/downcast/kernel failure.
    fn binary(&self, op: BinKind, a: &Dev, b: &Dev) -> Result<Dev, HandleError>;
    /// Elementwise unary (or transpose) on a resident array.
    ///
    /// # Errors
    /// [`HandleError::Backend`] on downcast/kernel failure.
    fn unary(&self, op: UnaryKind, a: &Dev) -> Result<Dev, HandleError>;
    /// Softmax / reduction along `axis` (`None` = all elements).
    ///
    /// # Errors
    /// [`HandleError::Backend`] on a bad axis or kernel failure.
    fn axis_op(
        &self,
        op: AxisKind,
        a: &Dev,
        axis: Option<usize>,
        keep_dims: bool,
    ) -> Result<Dev, HandleError>;
    /// Reinterpret a resident array with new dims.
    ///
    /// # Errors
    /// [`HandleError::Backend`] on an element-count mismatch.
    fn reshape(&self, a: &Dev, dims: &[usize]) -> Result<Dev, HandleError>;
    /// A resident array of `dims` filled with `value` (`dims = []`
    /// is a broadcastable scalar) -- backward passes use it for
    /// gradient seeds and fill-style formulas.
    ///
    /// # Errors
    /// [`HandleError::Backend`] on allocation failure.
    fn full(&self, dims: &[usize], value: f64) -> Result<Dev, HandleError>;
    /// Fused cross-entropy forward: mean over rows of
    /// `logsumexp(logits[i, :]) - logits[i, targets[i]]`.
    ///
    /// # Errors
    /// [`HandleError::Backend`] on shape/target-range failures.
    fn cross_entropy(&self, logits: &Dev, targets: &[usize]) -> Result<Dev, HandleError>;
}
