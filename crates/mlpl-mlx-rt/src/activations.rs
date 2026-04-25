//! MLX-backed unary activations (Saga 14 step 002).
//!
//! Forward-pass only. Each op is a 1:1 elementwise map, so shape
//! and labels pass through unchanged. Backward passes (gradcheck
//! parity) land in step 006.

use mlpl_array::DenseArray;
use mlx_rs::Array as MlxArray;

use crate::common::{dense_to_mlx, mlx_to_dense_data};

/// `exp(x)` elementwise. Infallible (shape and labels are preserved
/// from `a`, so the MLX kernel only sees pre-validated input).
#[must_use]
pub fn exp(a: &DenseArray) -> DenseArray {
    apply_unary(a, |x| x.exp().expect("mlx exp on validated shape"))
}

/// Natural log `ln(x)` elementwise.
#[must_use]
pub fn log(a: &DenseArray) -> DenseArray {
    apply_unary(a, |x| x.log().expect("mlx log on validated shape"))
}

/// Hyperbolic tangent `tanh(x)` elementwise.
#[must_use]
pub fn tanh(a: &DenseArray) -> DenseArray {
    apply_unary(a, |x| {
        mlx_rs::ops::tanh(x).expect("mlx tanh on validated shape")
    })
}

/// Logistic sigmoid `1 / (1 + exp(-x))` elementwise.
#[must_use]
pub fn sigmoid(a: &DenseArray) -> DenseArray {
    apply_unary(a, |x| {
        mlx_rs::ops::sigmoid(x).expect("mlx sigmoid on validated shape")
    })
}

/// Rectified linear unit `max(x, 0)` elementwise.
#[must_use]
pub fn relu(a: &DenseArray) -> DenseArray {
    apply_unary(a, |x| {
        mlx_rs::nn::relu(x).expect("mlx relu on validated shape")
    })
}

/// Shared dispatch for the unary activations: cast to MLX fp32,
/// run `op`, cast back to f64, and re-attach shape and labels.
/// Output shape is identical to input, so no shape validation is
/// possible (or needed) -- DenseArray::new + with_labels cannot
/// fail on data the input already carried.
fn apply_unary(a: &DenseArray, op: impl FnOnce(&MlxArray) -> MlxArray) -> DenseArray {
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    let result = op(&mlx);
    let data = mlx_to_dense_data(result);
    let array =
        DenseArray::new(a.shape().clone(), data).expect("output element count matches input");
    match a.labels() {
        Some(lbls) => array
            .with_labels(lbls.to_vec())
            .expect("labels already validated on input"),
        None => array,
    }
}
