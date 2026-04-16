//! Shape and reduction primitives (Saga 14 step 002 refactor of lib.rs).
//!
//! Kept in its own module so `lib.rs` stays under the sw-checklist
//! function-count budget while Saga 14 grows the runtime surface.

use mlpl_array::{ArrayError, DenseArray, Shape};

/// `reshape(a, dims)` -- reinterpret `a` with new dims. Element
/// count must match or `ArrayError::ShapeMismatch` is returned.
/// Labels drop (Saga 11.5 semantics -- a reshaped array no longer
/// carries the old per-axis identity).
pub fn reshape(a: &DenseArray, dims: &[usize]) -> Result<DenseArray, ArrayError> {
    a.reshape(Shape::new(dims.to_vec()))
}

/// `transpose(a)` -- reverse axis order. Preserves Saga 11.5 labels
/// (labels reverse alongside axes).
#[must_use]
pub fn transpose(a: &DenseArray) -> DenseArray {
    a.transpose()
}

/// `reduce_add(a)` -- flat sum of all elements as a scalar.
#[must_use]
pub fn reduce_add(a: &DenseArray) -> DenseArray {
    let sum: f64 = a.data().iter().copied().sum();
    DenseArray::from_scalar(sum)
}

/// `reduce_add_axis(a, axis)` -- sum along `axis`, dropping that
/// dim (and its label, if any) from the result shape.
pub fn reduce_add_axis(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
    a.reduce_axis(axis, 0.0, |x, y| x + y)
}
