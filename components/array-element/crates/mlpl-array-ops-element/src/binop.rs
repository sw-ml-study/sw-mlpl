use mlpl_array::{ArrayError, DenseArray};

use crate::broadcast::{broadcast_apply, broadcast_shape};
use crate::merge_labels::merge_labels;

/// Validate that two arrays are compatible for an element-wise binary op --
/// broadcastable shapes AND unifiable labels -- WITHOUT computing the result.
/// Callers that build an op on a value graph rather than eagerly (the autograd
/// tape) use this to turn an incompatible shape or label into a clean error
/// instead of a panic (findings F10 / F18).
pub fn check_binop_compat(a: &DenseArray, b: &DenseArray) -> Result<(), ArrayError> {
    merge_labels(a, b)?;
    if a.rank() != 0 && b.rank() != 0 {
        broadcast_shape(a.shape().dims(), b.shape().dims())?;
    }
    Ok(())
}

/// Apply-binop extension for `DenseArray`.
pub trait ApplyBinopExt {
    /// Apply a binary op element-wise with NumPy / APL trailing-axis
    /// broadcasting: shapes align from the RIGHT, and any axis that is
    /// missing on one operand or has extent 1 broadcasts against the
    /// other. So `[2] * [1,2,3]` is `[2,4,6]` (single-element broadcast),
    /// and a rank-3 kernel `[C,kh,kw]` multiplies rank-5 patches
    /// `[oy,ox,C,kh,kw]` directly -- no reshape. Labels align from the
    /// right too (Saga 11.5 Phase 3 semantics), so the surviving axes of
    /// a labeled operand carry through.
    fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError>;
}

impl ApplyBinopExt for DenseArray {
    fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let labels = merge_labels(self, other)?;
        let (data, shape) = broadcast_apply(self, other, op)?;
        let arr = DenseArray::new(shape, data)?;
        match labels {
            Some(l) => arr.with_labels(l),
            None => Ok(arr),
        }
    }
}
