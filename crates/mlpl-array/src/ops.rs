//! Array operations: reshape, transpose, element-wise arithmetic.

use crate::dense::DenseArray;
use crate::error::ArrayError;
use crate::shape::Shape;

impl DenseArray {
    /// Reshape to a new shape, preserving element order.
    ///
    /// Succeeds only when the new shape has the same element count.
    pub fn reshape(&self, new_shape: Shape) -> Result<DenseArray, ArrayError> {
        let source = self.elem_count();
        let target = new_shape.elem_count();
        if source != target {
            return Err(ArrayError::ShapeMismatch { source, target });
        }
        Ok(DenseArray {
            shape: new_shape,
            data: self.data.clone(),
        })
    }

    /// Transpose: reverse axis order and reorder data to row-major.
    ///
    /// - Scalar/vector: returns a clone (identity).
    /// - Matrix and higher: reverses dims and physically reorders data.
    #[must_use]
    pub fn transpose(&self) -> DenseArray {
        let dims = self.shape().dims();
        if dims.len() <= 1 {
            return self.clone();
        }

        let new_dims: Vec<usize> = dims.iter().rev().copied().collect();
        let new_shape = Shape::new(new_dims);
        let n = self.elem_count();
        let mut new_data = vec![0.0; n];

        let rank = dims.len();
        let old_strides = compute_strides(dims);
        let new_strides = compute_strides(new_shape.dims());

        for flat in 0..n {
            let mut remainder = flat;
            let mut new_flat = 0;
            for axis in 0..rank {
                let idx = remainder / old_strides[axis];
                remainder %= old_strides[axis];
                new_flat += idx * new_strides[rank - 1 - axis];
            }
            new_data[new_flat] = self.data[flat];
        }

        DenseArray {
            shape: new_shape,
            data: new_data,
        }
    }
}

impl DenseArray {
    /// Apply a binary operation element-wise with scalar broadcasting.
    ///
    /// - Same shape: element-wise.
    /// - One scalar: broadcast to the other's shape.
    /// - Otherwise: ShapeMismatch error.
    pub fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        if self.shape() == other.shape() {
            let data: Vec<f64> = self
                .data()
                .iter()
                .zip(other.data().iter())
                .map(|(a, b)| op(*a, *b))
                .collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
            });
        }
        // Scalar broadcast
        if self.rank() == 0 {
            let s = self.data()[0];
            let data: Vec<f64> = other.data().iter().map(|b| op(s, *b)).collect();
            return Ok(DenseArray {
                shape: other.shape.clone(),
                data,
            });
        }
        if other.rank() == 0 {
            let s = other.data()[0];
            let data: Vec<f64> = self.data().iter().map(|a| op(*a, s)).collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
            });
        }
        Err(ArrayError::ShapeMismatch {
            source: self.elem_count(),
            target: other.elem_count(),
        })
    }
}

impl DenseArray {
    /// Dot product of two rank-1 vectors.
    ///
    /// Both must be vectors of the same length. Returns a scalar.
    pub fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(ArrayError::RankMismatch {
                expected: 1,
                got: if self.rank() != 1 {
                    self.rank()
                } else {
                    other.rank()
                },
            });
        }
        if self.elem_count() != other.elem_count() {
            return Err(ArrayError::ShapeMismatch {
                source: self.elem_count(),
                target: other.elem_count(),
            });
        }
        let sum: f64 = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(DenseArray::from_scalar(sum))
    }
}

impl DenseArray {
    /// Reduce along an axis using the given binary operation.
    ///
    /// Removes the specified axis from the shape. For example,
    /// a [2,3] array reduced along axis 0 produces a [3] result.
    pub fn reduce_axis(
        &self,
        axis: usize,
        identity: f64,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        if axis >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis,
                index: axis,
                size: dims.len(),
            });
        }
        let mut result_dims: Vec<usize> = dims.to_vec();
        result_dims.remove(axis);
        let result_shape = Shape::new(result_dims);
        let result_count = result_shape.elem_count();
        let mut result_data = vec![identity; result_count];

        let strides = compute_strides(dims);
        let axis_size = dims[axis];
        let axis_stride = strides[axis];

        for flat in 0..self.elem_count() {
            let result_flat = if axis_stride > 1 {
                let outer = flat / (axis_size * axis_stride);
                let inner = flat % axis_stride;
                outer * axis_stride + inner
            } else {
                flat / axis_size
            };
            result_data[result_flat] = op(result_data[result_flat], self.data[flat]);
        }

        Ok(DenseArray {
            shape: result_shape,
            data: result_data,
        })
    }
}

/// Compute strides for row-major layout.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}
