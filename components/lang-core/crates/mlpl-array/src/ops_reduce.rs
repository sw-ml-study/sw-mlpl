//! Saga 33 step 005: axis-reduction ops (`reduce_axis`,
//! `argmax_axis`) extracted from `ops.rs`. Both walk the row-major
//! buffer in flat-index order and use `compute_strides` to recover
//! the per-axis index.

use crate::dense::DenseArray;
use crate::error::ArrayError;
use crate::ops_strides::compute_strides;
use crate::shape::Shape;

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

        let labels = self.labels.as_ref().map(|lbls| {
            let mut out = lbls.clone();
            out.remove(axis);
            out
        });
        Ok(DenseArray {
            shape: result_shape,
            data: result_data,
            labels,
        })
    }

    /// Argmax along an axis. Returns an array with the given axis
    /// removed whose values are the indices (as f64) of the maxima.
    /// Ties go to the first occurrence.
    pub fn argmax_axis(&self, axis: usize) -> Result<DenseArray, ArrayError> {
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
        let mut best_val = vec![f64::NEG_INFINITY; result_count];
        let mut best_idx = vec![0.0f64; result_count];

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
            let axis_idx = (flat / axis_stride) % axis_size;
            let v = self.data()[flat];
            if v > best_val[result_flat] {
                best_val[result_flat] = v;
                best_idx[result_flat] = axis_idx as f64;
            }
        }

        let mut out = DenseArray::new(result_shape, best_idx)?;
        if let Some(lbls) = self.labels.as_ref() {
            let mut new_lbls = lbls.clone();
            new_lbls.remove(axis);
            out.labels = Some(new_lbls);
        }
        Ok(out)
    }
}
