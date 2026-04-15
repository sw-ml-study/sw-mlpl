//! Dense array storage for MLPL.

use crate::error::ArrayError;
use crate::shape::Shape;

/// A dense array with row-major contiguous storage.
///
/// `labels` is `Some` when axis names have been attached via the
/// Saga 11.5 `label(x, [...])` built-in or annotation syntax; fresh
/// arrays and every legacy construction path leave it `None`, which
/// means "positional" for every axis. An explicit inner `None` at a
/// given axis lets a partially-labeled shape (e.g.
/// `[None, Some("d_model")]`) keep its rank.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseArray {
    pub(crate) shape: Shape,
    pub(crate) data: Vec<f64>,
    pub(crate) labels: Option<Vec<Option<String>>>,
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
        Ok(Self {
            shape,
            data,
            labels: None,
        })
    }

    /// Create a zero-filled array with the given shape.
    #[must_use]
    pub fn zeros(shape: Shape) -> Self {
        let len = shape.elem_count();
        Self {
            shape,
            data: vec![0.0; len],
            labels: None,
        }
    }

    /// Create a rank-0 (scalar) array.
    #[must_use]
    pub fn from_scalar(value: f64) -> Self {
        Self {
            shape: Shape::scalar(),
            data: vec![value],
            labels: None,
        }
    }

    /// Create a rank-1 (vector) array.
    #[must_use]
    pub fn from_vec(data: Vec<f64>) -> Self {
        let shape = Shape::vector(data.len());
        Self {
            shape,
            data,
            labels: None,
        }
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

    /// Apply a function to every element, returning a new array.
    ///
    /// Labels propagate unchanged (Saga 11.5): `map` is 1:1 with
    /// identical shape, so per-axis identity is preserved.
    #[must_use]
    pub fn map(&self, f: fn(f64) -> f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
            labels: self.labels.clone(),
        }
    }
}
