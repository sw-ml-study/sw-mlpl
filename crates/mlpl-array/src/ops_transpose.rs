//! Saga 33 step 005: `transpose` extracted from `ops.rs`. Reverses
//! axis order and physically reorders the row-major buffer.

use crate::dense::DenseArray;
use crate::ops_strides::compute_strides;
use crate::shape::Shape;

impl DenseArray {
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

        let labels = self
            .labels
            .as_ref()
            .map(|lbls| lbls.iter().rev().cloned().collect());

        DenseArray {
            shape: new_shape,
            data: new_data,
            labels,
        }
    }
}
