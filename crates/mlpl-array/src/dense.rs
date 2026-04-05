//! Dense array storage for MLPL.

use crate::error::ArrayError;
use crate::shape::Shape;

/// A dense array with row-major contiguous storage.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseArray {
    pub(crate) shape: Shape,
    pub(crate) data: Vec<f64>,
}

impl DenseArray {
    /// Create from a shape and data vector.
    ///
    /// Returns `DataLengthMismatch` if lengths disagree.
    pub fn new(shape: Shape, data: Vec<f64>) -> Result<Self, ArrayError> {
        let expected = shape.elem_count();
        if data.len() != expected {
            return Err(ArrayError::DataLengthMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Create a zero-filled array with the given shape.
    #[must_use]
    pub fn zeros(shape: Shape) -> Self {
        let len = shape.elem_count();
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    /// Create a rank-0 (scalar) array.
    #[must_use]
    pub fn from_scalar(value: f64) -> Self {
        Self {
            shape: Shape::scalar(),
            data: vec![value],
        }
    }

    /// Create a rank-1 (vector) array.
    #[must_use]
    pub fn from_vec(data: Vec<f64>) -> Self {
        let shape = Shape::vector(data.len());
        Self { shape, data }
    }

    /// Borrow the shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Borrow the flat data slice.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}
