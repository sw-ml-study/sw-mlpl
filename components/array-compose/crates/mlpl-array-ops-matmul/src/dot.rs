use mlpl_array::{ArrayError, DenseArray};

/// Dot-product extension for `DenseArray`. Bring into scope with
/// `use mlpl_array_ops_matmul::prelude::*;` to call `a.dot(&b)`.
pub trait DotExt {
    /// Dot product of two rank-1 vectors of equal length. Returns a scalar.
    fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError>;
}

impl DotExt for DenseArray {
    fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        check_vectors(self, other)?;
        let s: f64 = self
            .data()
            .iter()
            .zip(other.data())
            .map(|(a, b)| a * b)
            .sum();
        Ok(DenseArray::from_scalar(s))
    }
}

/// Both operands must be rank-1 vectors of equal length.
fn check_vectors(a: &DenseArray, b: &DenseArray) -> Result<(), ArrayError> {
    if a.rank() != 1 || b.rank() != 1 {
        return Err(ArrayError::RankMismatch {
            expected: 1,
            got: a.rank().max(b.rank()),
        });
    }
    if a.elem_count() != b.elem_count() {
        return Err(ArrayError::ShapeMismatch {
            source: a.elem_count(),
            target: b.elem_count(),
        });
    }
    Ok(())
}
