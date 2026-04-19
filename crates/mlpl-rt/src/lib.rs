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
//!
//! Forward-pass primitives are split across sibling modules
//! (`elementwise`, `activations`, `transforms`, `reductions`) to
//! stay under the sw-checklist function-count budget as Saga 14
//! grows the surface. `mlpl-mlx` mirrors this layout and signature,
//! so the compile-to-rust codegen can target either runtime
//! interchangeably.

mod activations;
mod array_lit;
mod elementwise;
mod error;
mod reductions;
mod transforms;

pub use activations::{exp, log, relu, sigmoid, tanh};
pub use array_lit::array_lit;
pub use elementwise::{add, div, mul, neg, sub};
pub use mlpl_array::{ArrayError, DenseArray, Shape};
pub use mlpl_core::LabeledShape;
pub use reductions::{argmax, cross_entropy, log_softmax, mean, reduce_mul, softmax};
pub use transforms::{reduce_add, reduce_add_axis, reshape, transpose};

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
