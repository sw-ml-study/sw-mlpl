use mlpl_array::{ArrayError, DenseArray};

use crate::merge_labels::merge_labels;

/// Apply-binop extension for `DenseArray`.
pub trait ApplyBinopExt {
    /// Apply a binary op element-wise with scalar broadcasting.
    /// Labels propagate per Saga 11.5 Phase 3 semantics.
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
        } else if self.rank() == 0 {
            let s = self.data()[0];
            (
                other.data().iter().map(|b| op(s, *b)).collect(),
                other.shape().clone(),
            )
        } else if other.rank() == 0 {
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
