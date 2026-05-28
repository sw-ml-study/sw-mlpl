use mlpl_array::{ArrayError, DenseArray};

use crate::axis::{axis_walk, finish, project};

/// Reduce-along-axis extension for `DenseArray`. Bring into scope
/// with `use mlpl_array_ops_reduce::prelude::*;` to call
/// `a.reduce_axis(0, 0.0, |a, b| a + b)`.
pub trait ReduceAxisExt {
    /// Reduce along `axis` using `op` starting from `identity`.
    /// Removes `axis` from the shape.
    fn reduce_axis(
        &self,
        axis: usize,
        identity: f64,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError>;
}

impl ReduceAxisExt for DenseArray {
    fn reduce_axis(
        &self,
        axis: usize,
        identity: f64,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let w = axis_walk(self, axis)?;
        let mut data = vec![identity; w.result_shape.elem_count()];
        for (flat, &v) in self.data().iter().enumerate() {
            let r = project(flat, &w);
            data[r] = op(data[r], v);
        }
        finish(data, w)
    }
}
