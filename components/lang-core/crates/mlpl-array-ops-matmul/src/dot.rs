use mlpl_array::{ArrayError, DenseArray};

/// Dot-product extension for `DenseArray`. Bring into scope with
/// `use mlpl_array_ops_matmul::prelude::*;` to call `a.dot(&b)`.
pub trait DotExt {
    /// Dot product of two rank-1 vectors of equal length. Returns a scalar.
    fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError>;
}

impl DotExt for DenseArray {
    fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(ArrayError::RankMismatch {
                expected: 1,
                got: self.rank().max(other.rank()),
            });
        }
        if self.elem_count() != other.elem_count() {
            return Err(ArrayError::ShapeMismatch {
                source: self.elem_count(),
                target: other.elem_count(),
            });
        }
        let s: f64 = self
            .data()
            .iter()
            .zip(other.data())
            .map(|(a, b)| a * b)
            .sum();
        Ok(DenseArray::from_scalar(s))
    }
}
