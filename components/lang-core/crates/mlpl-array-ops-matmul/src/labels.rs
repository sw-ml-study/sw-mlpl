use mlpl_array::{ArrayError, DenseArray};

/// Compute the result labels for `matmul(a, b)`. The contraction
/// axis is `a`'s last dim vs `b`'s first dim; if both sides name it
/// and the names differ, raise `LabelMismatch`. Output labels are
/// the non-contracted dims. Saga 11.5 Phase 3 semantics, preserved
/// verbatim from the original mlpl-array implementation.
pub(crate) fn matmul_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.labels().is_none() && b.labels().is_none() {
        return Ok(None);
    }
    let default_b = vec![None; b.rank()];
    let al: &[Option<String>] = a.labels().unwrap_or(&[None, None]);
    let bl: &[Option<String>] = b.labels().unwrap_or(default_b.as_slice());
    if let (Some(sa), Some(sb)) = (&al[1], &bl[0])
        && sa != sb
    {
        let (expected, actual) = (al.to_vec(), bl.to_vec());
        return Err(ArrayError::LabelMismatch { expected, actual });
    }
    let mut result = vec![al[0].clone()];
    if b.rank() == 2 {
        result.push(bl[1].clone());
    }
    Ok(Some(result))
}
