//! Shared plumbing for the MLX-backed primitives.
//!
//! Two classes of helper live here:
//!
//! 1. **Conversion** -- `dense_to_mlx` / `mlx_to_dense` round-trip
//!    an `mlpl-array` `DenseArray` through an `mlx_rs::Array` on
//!    the fp32 GPU/Accelerate path. Every op that touches MLX
//!    goes through these so the f64 <-> f32 boundary (and the
//!    `.eval()` call that materializes MLX's lazy graph) lives in
//!    exactly one place.
//!
//! 2. **Label propagation** -- `merge_labels` / `matmul_labels`
//!    mirror `mlpl-array`'s private helpers of the same names
//!    byte-for-byte (Saga 11.5 Phase 3 semantics). They live here
//!    so the MLX ops that need them (elementwise, matmul) can
//!    share a single source of truth and `mlpl-array`'s internal
//!    helpers stay internal.

use mlpl_array::{ArrayError, DenseArray, Shape};
use mlx_rs::Array as MlxArray;

/// Build an MLX fp32 array from an `mlpl-array` `DenseArray`.
///
/// The caller is responsible for any shape validation -- this
/// helper trusts `dims` and only handles the f64 -> f32 cast plus
/// the i32 dims conversion MLX expects.
pub(crate) fn dense_to_mlx(data: &[f64], dims: &[usize]) -> MlxArray {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let shape: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    MlxArray::from_slice(&data_f32, &shape)
}

/// Materialize an MLX array and cast its flat contents back to f64.
///
/// `.eval()` forces MLX's lazy graph so `as_slice` sees real data.
/// The caller decides what shape (and labels) the resulting
/// `DenseArray` wears.
pub(crate) fn mlx_to_dense_data(mlx: MlxArray) -> Vec<f64> {
    mlx.eval()
        .expect("mlx eval on pre-validated shapes should not fail");
    let out: &[f32] = mlx.as_slice();
    out.iter().map(|&x| x as f64).collect()
}

/// Finalize an MLX computation: wrap `data` in a `DenseArray` of
/// `shape` and attach `labels` if present. Consolidates the
/// DenseArray::new + with_labels dance every op does on the way out.
pub(crate) fn finalize(
    shape: Shape,
    data: Vec<f64>,
    labels: Option<Vec<Option<String>>>,
) -> Result<DenseArray, ArrayError> {
    let array = DenseArray::new(shape, data)?;
    match labels {
        Some(lbls) => array.with_labels(lbls),
        None => Ok(array),
    }
}

/// Mirror of `mlpl-array`'s private `merge_labels` helper.
///
/// Scalars contribute no labels, so the non-scalar side's labels
/// win unconditionally when one operand is rank 0. For non-scalar
/// pairs, two unlabeled sides stay unlabeled, a single labeled
/// side carries its labels through, and two labeled sides must
/// agree or `LabelMismatch` is returned.
pub(crate) fn merge_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.rank() == 0 {
        return Ok(b.labels().map(<[Option<String>]>::to_vec));
    }
    if b.rank() == 0 {
        return Ok(a.labels().map(<[Option<String>]>::to_vec));
    }
    match (a.labels(), b.labels()) {
        (None, None) => Ok(None),
        (Some(l), None) | (None, Some(l)) => Ok(Some(l.to_vec())),
        (Some(la), Some(lb)) if la == lb => Ok(Some(la.to_vec())),
        (Some(la), Some(lb)) => Err(ArrayError::LabelMismatch {
            expected: la.to_vec(),
            actual: lb.to_vec(),
        }),
    }
}

/// Mirror of `mlpl-array`'s private `matmul_labels` helper.
/// Saga 11.5 Phase 3 semantics: contraction-axis labels must
/// agree, survivors are `[a.labels[0], b.labels[1]]`.
pub(crate) fn matmul_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.labels().is_none() && b.labels().is_none() {
        return Ok(None);
    }
    let default_b = vec![None; b.rank()];
    let al: &[Option<String>] = a.labels().unwrap_or(&[None, None][..]);
    let bl: &[Option<String>] = b.labels().unwrap_or(default_b.as_slice());
    if let (Some(sa), Some(sb)) = (&al[1], &bl[0])
        && sa != sb
    {
        return Err(ArrayError::LabelMismatch {
            expected: al.to_vec(),
            actual: bl.to_vec(),
        });
    }
    let mut result = vec![al[0].clone()];
    if b.rank() == 2 {
        result.push(bl[1].clone());
    }
    Ok(Some(result))
}
