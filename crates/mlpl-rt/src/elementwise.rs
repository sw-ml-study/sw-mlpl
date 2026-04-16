//! Elementwise binary primitives (Saga 14 step 002).
//!
//! Thin wrappers over `DenseArray::apply_binop` that let compiled
//! MLPL code call `mlpl_rt::add(&a, &b)` instead of a trait method.
//! The `mlpl-mlx` sibling crate exposes the same signatures so the
//! compile-to-rust codegen can target either runtime without source
//! changes.
//!
//! Label propagation is inherited from `apply_binop` -- the
//! non-scalar side's labels win, or matching label vectors
//! pass through, or `LabelMismatch` is raised.

use mlpl_array::{ArrayError, DenseArray};

/// Elementwise `a + b` with scalar broadcasting.
pub fn add(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    a.apply_binop(b, |x, y| x + y)
}

/// Elementwise `a - b` with scalar broadcasting.
pub fn sub(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    a.apply_binop(b, |x, y| x - y)
}

/// Elementwise `a * b` (Hadamard product) with scalar broadcasting.
pub fn mul(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    a.apply_binop(b, |x, y| x * y)
}

/// Elementwise `a / b` with scalar broadcasting.
pub fn div(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    a.apply_binop(b, |x, y| x / y)
}

/// Unary elementwise negation. Labels propagate unchanged.
#[must_use]
pub fn neg(a: &DenseArray) -> DenseArray {
    a.map(|x| -x)
}
