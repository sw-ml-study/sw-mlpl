//! Shared shape/label prep + GPU dispatch for the binary
//! elementwise ops. The public ops in `arith.rs` are one-line
//! wrappers over `binary`.

use candle_core::Tensor;
use mlpl_array::{ArrayError, DenseArray, Shape};
use mlpl_cuda_rt::{Labels, cuda_to_dense_data, dense_to_cuda, finalize};

/// Merge labels for a binary op: scalars defer to the non-scalar
/// side; two labeled non-scalar sides must agree.
///
/// # Errors
/// `LabelMismatch` if both sides are labeled and disagree.
fn merge_labels(a: &DenseArray, b: &DenseArray) -> Result<Labels, ArrayError> {
    match (a.rank(), b.rank()) {
        (0, _) => Ok(b.labels().map(<[Option<String>]>::to_vec)),
        (_, 0) => Ok(a.labels().map(<[Option<String>]>::to_vec)),
        _ => match (a.labels(), b.labels()) {
            (None, None) => Ok(None),
            (Some(l), None) | (None, Some(l)) => Ok(Some(l.to_vec())),
            (Some(la), Some(lb)) if la == lb => Ok(Some(la.to_vec())),
            (Some(la), Some(lb)) => Err(ArrayError::LabelMismatch {
                expected: la.to_vec(),
                actual: lb.to_vec(),
            }),
        },
    }
}

/// Resolve the result shape for a binary op: scalar broadcast or
/// same-shape equality.
///
/// # Errors
/// `ShapeMismatch` if neither operand is scalar and shapes differ.
fn result_shape(a: &DenseArray, b: &DenseArray) -> Result<Shape, ArrayError> {
    if a.rank() == 0 {
        Ok(b.shape().clone())
    } else if b.rank() == 0 || a.shape() == b.shape() {
        Ok(a.shape().clone())
    } else {
        Err(ArrayError::ShapeMismatch {
            source: a.elem_count(),
            target: b.elem_count(),
        })
    }
}

/// Shared binary dispatch: prep shape/labels, move both operands to
/// the GPU, run `op`, finalize.
///
/// # Errors
/// Propagates label/shape errors and any from `finalize`.
pub(crate) fn binary(
    a: &DenseArray,
    b: &DenseArray,
    op: impl FnOnce(&Tensor, &Tensor) -> Tensor,
) -> Result<DenseArray, ArrayError> {
    let labels = merge_labels(a, b)?;
    let shape = result_shape(a, b)?;
    let at = dense_to_cuda(a.data(), a.shape().dims());
    let bt = dense_to_cuda(b.data(), b.shape().dims());
    finalize(shape, cuda_to_dense_data(&op(&at, &bt)), labels)
}
