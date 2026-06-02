//! Binary elementwise arithmetic with scalar broadcasting. candle's
//! `broadcast_*` ops cover both the same-shape and scalar-broadcast
//! cases the CPU path allows; dispatch goes through `dispatch::binary`.

use crate::dispatch::binary;
use mlpl_array::{ArrayError, DenseArray};

/// Elementwise `a + b` with scalar broadcasting.
///
/// # Errors
/// `ShapeMismatch`/`LabelMismatch` for incompatible operands.
///
/// # Panics
/// Panics if the candle kernel fails on pre-validated shapes.
pub fn add(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    binary(a, b, |x, y| {
        x.broadcast_add(y).expect("cuda add on validated shapes")
    })
}

/// Elementwise `a - b` with scalar broadcasting.
///
/// # Errors
/// `ShapeMismatch`/`LabelMismatch` for incompatible operands.
///
/// # Panics
/// Panics if the candle kernel fails on pre-validated shapes.
pub fn sub(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    binary(a, b, |x, y| {
        x.broadcast_sub(y).expect("cuda sub on validated shapes")
    })
}

/// Elementwise `a * b` (Hadamard) with scalar broadcasting.
///
/// # Errors
/// `ShapeMismatch`/`LabelMismatch` for incompatible operands.
///
/// # Panics
/// Panics if the candle kernel fails on pre-validated shapes.
pub fn mul(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    binary(a, b, |x, y| {
        x.broadcast_mul(y).expect("cuda mul on validated shapes")
    })
}

/// Elementwise `a / b` with scalar broadcasting.
///
/// # Errors
/// `ShapeMismatch`/`LabelMismatch` for incompatible operands.
///
/// # Panics
/// Panics if the candle kernel fails on pre-validated shapes.
pub fn div(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    binary(a, b, |x, y| {
        x.broadcast_div(y).expect("cuda div on validated shapes")
    })
}
