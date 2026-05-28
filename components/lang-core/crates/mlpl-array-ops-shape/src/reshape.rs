use mlpl_array::{ArrayError, DenseArray, Shape};

/// Reshape extension for `DenseArray`.
pub trait ReshapeExt {
    /// Reshape to a new shape, preserving element order. Succeeds
    /// only when the new shape has the same element count.
    fn reshape(&self, new_shape: Shape) -> Result<DenseArray, ArrayError>;
}

impl ReshapeExt for DenseArray {
    fn reshape(&self, new_shape: Shape) -> Result<DenseArray, ArrayError> {
        let (source, target) = (self.elem_count(), new_shape.elem_count());
        if source != target {
            return Err(ArrayError::ShapeMismatch { source, target });
        }
        DenseArray::new(new_shape, self.data().to_vec())
    }
}
