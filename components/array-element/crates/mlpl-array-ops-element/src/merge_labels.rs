use mlpl_array::{ArrayError, DenseArray};

/// Compute the label list for an element-wise op result.
///
/// Scalars contribute no labels (the non-scalar side wins). Same-rank
/// pairs combine PER AXIS: two unlabeled stay unlabeled, one labeled
/// carries through, and a `None` (unlabeled) axis unifies with the
/// other operand's label as a wildcard -- only two DIFFERENT explicit
/// labels on the same axis are a `LabelMismatch`. This matches the
/// different-rank broadcast path (`merge_broadcast_labels`), so e.g. a
/// labeled `[time, dim]` table adds to a partially-labeled `[time, _]`
/// tensor (a positional encoding onto an embedding on the autograd
/// tape) instead of panicking. Saga 11.5 Phase 3 semantics, relaxed
/// from whole-vector equality to per-axis unification.
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
    if a.rank() != b.rank() {
        return merge_broadcast_labels(a, b);
    }
    match (a.labels(), b.labels()) {
        (None, None) => Ok(None),
        (Some(l), None) | (None, Some(l)) => Ok(Some(l.to_vec())),
        (Some(la), Some(lb)) => unify_axis_labels(la, lb),
    }
}

/// Unify two same-length per-axis label lists: a `None` axis takes the
/// other's label; two explicit labels must agree or it is a
/// `LabelMismatch`.
fn unify_axis_labels(
    la: &[Option<String>],
    lb: &[Option<String>],
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    let mut out = Vec::with_capacity(la.len());
    let mut any = false;
    for (x, y) in la.iter().zip(lb) {
        out.push(match (x, y) {
            (Some(x), Some(y)) if x != y => {
                return Err(ArrayError::LabelMismatch {
                    expected: la.to_vec(),
                    actual: lb.to_vec(),
                });
            }
            (Some(v), _) | (_, Some(v)) => {
                any = true;
                Some(v.clone())
            }
            (None, None) => None,
        });
    }
    Ok(any.then_some(out))
}

/// Labels for a broadcast between DIFFERENT non-zero ranks: axes align
/// from the RIGHT (output rank = the larger). Each output axis takes the
/// label of whichever operand has it; if both label the same axis they
/// must agree. A prepended axis (present on only the wider operand)
/// keeps that operand's label.
fn merge_broadcast_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    let r = a.rank().max(b.rank());
    let label_at = |arr: &DenseArray, i: usize| -> Option<String> {
        let off = r - arr.rank();
        if i < off {
            return None;
        }
        arr.labels().and_then(|ls| ls[i - off].clone())
    };
    let mut out = vec![None; r];
    let mut any = false;
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = match (label_at(a, i), label_at(b, i)) {
            (Some(x), Some(y)) if x != y => {
                return Err(ArrayError::LabelMismatch {
                    expected: vec![Some(x)],
                    actual: vec![Some(y)],
                });
            }
            (Some(x), _) | (_, Some(x)) => {
                any = true;
                Some(x)
            }
            (None, None) => None,
        };
    }
    Ok(any.then_some(out))
}
