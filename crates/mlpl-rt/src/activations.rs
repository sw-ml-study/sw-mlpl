//! Unary activation primitives (Saga 14 step 002).
//!
//! Forward-pass only. Backward passes land in step 006 alongside
//! the autograd tape. Labels propagate unchanged -- each of these
//! is a 1:1 elementwise map, so per-axis identity is preserved.

use mlpl_array::DenseArray;

/// `exp(x)` elementwise.
#[must_use]
pub fn exp(a: &DenseArray) -> DenseArray {
    a.map(f64::exp)
}

/// Natural log `ln(x)` elementwise. Undefined on non-positive input.
#[must_use]
pub fn log(a: &DenseArray) -> DenseArray {
    a.map(f64::ln)
}

/// Hyperbolic tangent `tanh(x)` elementwise.
#[must_use]
pub fn tanh(a: &DenseArray) -> DenseArray {
    a.map(f64::tanh)
}

/// Logistic sigmoid `1 / (1 + exp(-x))` elementwise.
#[must_use]
pub fn sigmoid(a: &DenseArray) -> DenseArray {
    a.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Rectified linear unit `max(x, 0)` elementwise.
#[must_use]
pub fn relu(a: &DenseArray) -> DenseArray {
    a.map(|x| x.max(0.0))
}
