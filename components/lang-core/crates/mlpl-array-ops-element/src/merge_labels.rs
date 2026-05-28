use mlpl_array::{ArrayError, DenseArray};

/// Compute the label list for an element-wise op result.
///
/// Scalars contribute no labels (the non-scalar side wins). Same-rank
/// pairs combine: two unlabeled stay unlabeled, one labeled carries
/// through, two labeled must agree or `LabelMismatch`. Saga 11.5
/// Phase 3 semantics, preserved verbatim from the original mlpl-array
/// implementation.
pub(crate) fn merge_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.rank() == 0 {
        return Ok(b.labels().map(<[_]>::to_vec));
    }
    if b.rank() == 0 {
        return Ok(a.labels().map(<[_]>::to_vec));
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
