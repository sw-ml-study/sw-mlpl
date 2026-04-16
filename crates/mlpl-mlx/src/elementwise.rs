//! MLX-backed elementwise binary primitives (Saga 14 step 002).
//!
//! Signatures mirror `mlpl-rt`: each op takes two `&DenseArray`
//! references and returns `Result<DenseArray, ArrayError>`. Shape
//! validation -- rank-0 broadcasting and same-shape equality --
//! lives on the Rust side so MLX sees only legal inputs; label
//! propagation goes through the shared `merge_labels` helper that
//! mirrors `mlpl-array`'s private logic byte-for-byte.

use mlpl_array::{ArrayError, DenseArray};
use mlx_rs::Array as MlxArray;

use crate::common::{dense_to_mlx, finalize, merge_labels, mlx_to_dense_data};

/// Elementwise `a + b` with scalar broadcasting.
pub fn add(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    dispatch(a, b, |x, y| x.add(y).expect("mlx add on validated shapes"))
}

/// Elementwise `a - b` with scalar broadcasting.
pub fn sub(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    dispatch(a, b, |x, y| {
        x.subtract(y).expect("mlx subtract on validated shapes")
    })
}

/// Elementwise `a * b` (Hadamard product) with scalar broadcasting.
pub fn mul(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    dispatch(a, b, |x, y| {
        x.multiply(y).expect("mlx multiply on validated shapes")
    })
}

/// Elementwise `a / b` with scalar broadcasting.
pub fn div(a: &DenseArray, b: &DenseArray) -> Result<DenseArray, ArrayError> {
    dispatch(a, b, |x, y| {
        x.divide(y).expect("mlx divide on validated shapes")
    })
}

/// Unary elementwise negation. Infallible: output shape and labels
/// match the input, so every intermediate is pre-validated.
#[must_use]
pub fn neg(a: &DenseArray) -> DenseArray {
    let mlx = dense_to_mlx(a.data(), a.shape().dims());
    let result = mlx.negative().expect("mlx negative on validated shape");
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

/// Shared dispatch for the binary ops: validate shapes (same or
/// one-side-scalar), merge labels, run the MLX op, finalize. The
/// result shape is the non-scalar operand's shape; for same-shape
/// inputs it's either (they agree).
fn dispatch(
    a: &DenseArray,
    b: &DenseArray,
    op: impl FnOnce(&MlxArray, &MlxArray) -> MlxArray,
) -> Result<DenseArray, ArrayError> {
    let labels = merge_labels(a, b)?;
    let result_shape = if a.rank() == 0 {
        b.shape().clone()
    } else if b.rank() == 0 || a.shape() == b.shape() {
        a.shape().clone()
    } else {
        return Err(ArrayError::ShapeMismatch {
            source: a.elem_count(),
            target: b.elem_count(),
        });
    };
    let a_mlx = dense_to_mlx(a.data(), a.shape().dims());
    let b_mlx = dense_to_mlx(b.data(), b.shape().dims());
    let out = op(&a_mlx, &b_mlx);
    finalize(result_shape, mlx_to_dense_data(out), labels)
}
