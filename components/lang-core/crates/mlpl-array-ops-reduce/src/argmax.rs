use mlpl_array::{ArrayError, DenseArray};

use crate::axis::{axis_walk, finish, project};

/// Argmax-along-axis extension for `DenseArray`. Bring into scope
/// with `use mlpl_array_ops_reduce::prelude::*;` to call
/// `a.argmax_axis(0)`.
pub trait ArgmaxAxisExt {
    /// Argmax along `axis`. Result shape has `axis` removed; values
    /// are the indices (as f64) of the maxima. Ties go to the first.
    fn argmax_axis(&self, axis: usize) -> Result<DenseArray, ArrayError>;
}

impl ArgmaxAxisExt for DenseArray {
    fn argmax_axis(&self, axis: usize) -> Result<DenseArray, ArrayError> {
        let w = axis_walk(self, axis)?;
        let n = w.result_shape.elem_count();
        let mut best_val = vec![f64::NEG_INFINITY; n];
        let mut best_idx = vec![0.0f64; n];
        for (flat, &v) in self.data().iter().enumerate() {
            let r = project(flat, &w);
            if v > best_val[r] {
                best_val[r] = v;
                best_idx[r] = ((flat / w.axis_stride) % w.axis_size) as f64;
            }
        }
        finish(best_idx, w)
    }
}
