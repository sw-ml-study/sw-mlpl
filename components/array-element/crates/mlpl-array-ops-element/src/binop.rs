use mlpl_array::{ArrayError, DenseArray, Shape};

use crate::merge_labels::merge_labels;

/// Apply-binop extension for `DenseArray`.
pub trait ApplyBinopExt {
    /// Apply a binary op element-wise with single-element broadcasting:
    /// if one operand holds exactly one value (a rank-0 scalar OR a
    /// length-1 array like `[2]` / `[[2]]`), that value broadcasts
    /// against the other operand's shape, matching NumPy / APL. This
    /// means `[2] * [1,2,3]` is `[2,4,6]`, and an indexing primitive
    /// that returns a length-1 slice meets a vector without an explicit
    /// `reshape(..., [])` collapse. Labels propagate per Saga 11.5
    /// Phase 3 semantics.
    fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError>;
}

impl ApplyBinopExt for DenseArray {
    fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let labels = merge_labels(self, other)?;
        let (data, shape) = broadcast_apply(self, other, op)?;
        let arr = DenseArray::new(shape, data)?;
        match labels {
            Some(l) => arr.with_labels(l),
            None => Ok(arr),
        }
    }
}

/// Element-wise apply with single-element broadcasting -> `(data, shape)`
/// (labels are handled by the caller). Broadcasting PRESERVES the array
/// operand's rank: a single-element operand (rank-0 scalar OR length-1
/// array) broadcasts against the other's shape, and when BOTH are
/// single-element the higher-rank shape wins -- so an all-unit shape
/// survives scalar broadcast (`[[0]] * 1` is `[1, 1]`, not `[]`;
/// regression fix, demo-abstract-algebra BUG 1). Ranks can only differ
/// in that both-single case, since equal-rank single-element operands
/// share a shape and take the equal-shape branch.
fn broadcast_apply(
    a: &DenseArray,
    b: &DenseArray,
    op: fn(f64, f64) -> f64,
) -> Result<(Vec<f64>, Shape), ArrayError> {
    if a.shape() == b.shape() {
        Ok((zip_with(a.data(), b.data(), op), a.shape().clone()))
    } else if a.elem_count() == 1 && b.elem_count() == 1 {
        let shape = if a.rank() >= b.rank() {
            a.shape().clone()
        } else {
            b.shape().clone()
        };
        Ok((vec![op(a.data()[0], b.data()[0])], shape))
    } else if a.elem_count() == 1 {
        let s = a.data()[0];
        Ok((
            b.data().iter().map(|x| op(s, *x)).collect(),
            b.shape().clone(),
        ))
    } else if b.elem_count() == 1 {
        let s = b.data()[0];
        Ok((
            a.data().iter().map(|x| op(*x, s)).collect(),
            a.shape().clone(),
        ))
    } else {
        Err(ArrayError::ShapeMismatch {
            source: a.elem_count(),
            target: b.elem_count(),
        })
    }
}

fn zip_with(a: &[f64], b: &[f64], op: fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| op(*x, *y)).collect()
}
