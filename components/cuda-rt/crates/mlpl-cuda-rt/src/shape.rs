//! CUDA-backed shape primitives. `reshape` routes through candle
//! (dropping labels, like the CPU path); `transpose` delegates to
//! the CPU stride walk (cheap at these sizes; candle transpose is a
//! strided view whose flat readback would not match without an
//! explicit contiguous copy).

use crate::convert::{cuda_to_dense_data, dense_to_cuda, finalize};
use mlpl_array::{ArrayError, DenseArray, Shape};
use mlpl_array_ops_shape::prelude::*;

/// `reshape(a, dims)` -- reinterpret with new dims. Element count
/// must match. Labels drop, matching the CPU path.
///
/// # Errors
/// `ShapeMismatch` if the element count differs from `dims`'s product.
///
/// # Panics
/// Panics if the candle reshape fails on a pre-validated element count.
pub fn reshape(a: &DenseArray, dims: &[usize]) -> Result<DenseArray, ArrayError> {
    let source = a.elem_count();
    let target: usize = dims.iter().product();
    if source != target {
        return Err(ArrayError::ShapeMismatch { source, target });
    }
    let t = dense_to_cuda(a.data(), a.shape().dims());
    let r = t
        .reshape(dims.to_vec())
        .expect("cuda reshape on validated count");
    finalize(Shape::new(dims.to_vec()), cuda_to_dense_data(&r), None)
}

/// `transpose(a)` -- reverse axis order, reorder data to row-major,
/// reverse labels. Delegates to the CPU stride walk.
#[must_use]
pub fn transpose(a: &DenseArray) -> DenseArray {
    a.transpose()
}
