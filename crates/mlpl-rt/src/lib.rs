//! MLPL runtime target crate (Saga: compile-to-rust).
//!
//! Compiled MLPL code -- emitted by the future `mlpl!` proc macro
//! and the `mlpl build` subcommand -- calls into this crate. It
//! re-exports the value and shape types from `mlpl-array` and
//! `mlpl-core`, and provides typed Rust signatures for a handful of
//! core primitives so the codegen can emit straight-line calls like
//! `mlpl_rt::iota(5)` and `mlpl_rt::reshape(&x, &[2, 3])?`.
//!
//! This crate deliberately does *not* depend on `mlpl-eval`,
//! `mlpl-runtime`, or `mlpl-parser`. Those crates are tooling;
//! compiled binaries link only against `mlpl-rt` so they never
//! carry a parser or an interpreter at runtime.

pub use mlpl_array::{ArrayError, DenseArray, Shape};
pub use mlpl_core::LabeledShape;

/// `iota(n)` -- the vector `[0, 1, ..., n-1]` as f64 values.
#[must_use]
pub fn iota(n: usize) -> DenseArray {
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    DenseArray::from_vec(data)
}

/// `shape(a)` -- the dimension list of `a` as a rank-1 array.
#[must_use]
pub fn shape(a: &DenseArray) -> DenseArray {
    let dims: Vec<f64> = a.shape().dims().iter().map(|&d| d as f64).collect();
    DenseArray::from_vec(dims)
}

/// `rank(a)` -- the number of dimensions of `a` as a scalar.
#[must_use]
pub fn rank(a: &DenseArray) -> DenseArray {
    DenseArray::from_scalar(a.rank() as f64)
}

/// `reshape(a, dims)` -- reinterpret `a` with new dims; element count
/// must match or `ArrayError::ShapeMismatch` is returned.
pub fn reshape(a: &DenseArray, dims: &[usize]) -> Result<DenseArray, ArrayError> {
    a.reshape(Shape::new(dims.to_vec()))
}

/// `transpose(a)` -- reverse axis order. Preserves Saga 11.5 labels.
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
