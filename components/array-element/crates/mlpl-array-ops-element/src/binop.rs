use mlpl_array::{ArrayError, DenseArray};

use crate::merge_labels::merge_labels;

/// Apply-binop extension for `DenseArray`.
pub trait ApplyBinopExt {
    /// Apply a binary op element-wise with single-element broadcasting:
    /// if one operand holds exactly one value (a rank-0 scalar OR a
    /// length-1 array like `[2]` / `[[2]]`), that value broadcasts
    /// against the other operand's shape, matching NumPy / APL. This
    /// means `[2] * [1,2,3]` is `[2,4,6]`, and an indexing primitive
    /// that returns a length-1 slice meets a vector without an explicit
    /// `reshape(..., [])` collapse. Labels propagate per Saga 11.5
    /// Phase 3 semantics.
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
        let (data, shape) = if self.shape() == other.shape() {
            (
                zip_with(self.data(), other.data(), op),
                self.shape().clone(),
            )
        } else if self.elem_count() == 1 {
            // A single-element operand (rank-0 scalar OR a length-1
            // array) broadcasts its one value against the other shape.
            let s = self.data()[0];
            (
                other.data().iter().map(|b| op(s, *b)).collect(),
                other.shape().clone(),
            )
        } else if other.elem_count() == 1 {
            let s = other.data()[0];
            (
                self.data().iter().map(|a| op(*a, s)).collect(),
                self.shape().clone(),
            )
        } else {
            return Err(ArrayError::ShapeMismatch {
                source: self.elem_count(),
                target: other.elem_count(),
            });
        };
        let arr = DenseArray::new(shape, data)?;
        match labels {
            Some(l) => arr.with_labels(l),
            None => Ok(arr),
        }
    }
}

fn zip_with(a: &[f64], b: &[f64], op: fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| op(*x, *y)).collect()
}
