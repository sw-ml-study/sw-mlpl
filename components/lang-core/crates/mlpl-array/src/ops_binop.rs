//! Saga 33 step 005: element-wise binary operation with scalar
//! broadcasting + label propagation, extracted from `ops.rs`.

use crate::dense::DenseArray;
use crate::error::ArrayError;

impl DenseArray {
    /// Apply a binary operation element-wise with scalar broadcasting.
    ///
    /// - Same shape: element-wise.
    /// - One scalar: broadcast to the other's shape.
    /// - Otherwise: ShapeMismatch error.
    ///
    /// Label propagation (Saga 11.5 Phase 3): scalar operands contribute
    /// no labels, so the non-scalar side's labels win unconditionally.
    /// For same-shape operands: if either side is unlabeled, the
    /// labeled side's labels carry through; if both are labeled, the
    /// label vectors must match or `LabelMismatch` is raised.
    pub fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let result_labels = merge_labels(self, other)?;
        if self.shape() == other.shape() {
            let data: Vec<f64> = self
                .data()
                .iter()
                .zip(other.data().iter())
                .map(|(a, b)| op(*a, *b))
                .collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        // Scalar broadcast
        if self.rank() == 0 {
            let s = self.data()[0];
            let data: Vec<f64> = other.data().iter().map(|b| op(s, *b)).collect();
            return Ok(DenseArray {
                shape: other.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        if other.rank() == 0 {
            let s = other.data()[0];
            let data: Vec<f64> = self.data().iter().map(|a| op(*a, s)).collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        Err(ArrayError::ShapeMismatch {
            source: self.elem_count(),
            target: other.elem_count(),
        })
    }
}

/// Compute the label list for the result of an elementwise op on two
/// arrays. Scalars contribute no labels, so the non-scalar side wins
/// unconditionally when one operand is rank 0. For non-scalar pairs,
/// two unlabeled sides stay unlabeled, a single labeled side carries
/// its labels through, and two labeled sides must agree or
/// `LabelMismatch` is returned. Saga 11.5 Phase 3.
fn merge_labels(a: &DenseArray, b: &DenseArray) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.rank() == 0 {
        return Ok(b.labels.clone());
    }
    if b.rank() == 0 {
        return Ok(a.labels.clone());
    }
    match (&a.labels, &b.labels) {
        (None, None) => Ok(None),
        (Some(l), None) | (None, Some(l)) => Ok(Some(l.clone())),
        (Some(la), Some(lb)) if la == lb => Ok(Some(la.clone())),
        (Some(la), Some(lb)) => Err(ArrayError::LabelMismatch {
            expected: la.clone(),
            actual: lb.clone(),
        }),
    }
}
