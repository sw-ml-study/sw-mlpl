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

/// Compute strides for row-major layout.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}
