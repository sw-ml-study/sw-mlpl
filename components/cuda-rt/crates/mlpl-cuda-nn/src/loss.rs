//! CUDA-backed `cross_entropy`: a fused log-softmax + NLL over
//! `[N, V]` logits returning an unlabeled scalar mean, mirroring
//! `mlpl-rt`. The per-row log-sum-exp runs on the GPU; the gather +
//! mean run on the CPU. Responsibilities are split across small
//! helpers (shape resolution, target validation, the row LSE).

use mlpl_array::{ArrayError, DenseArray};
use mlpl_cuda_rt::{cuda_to_dense_data, dense_to_cuda};

/// `cross_entropy(logits, targets)`: scalar mean negative
/// log-likelihood of the `targets` class under `softmax(logits)`.
///
/// # Errors
/// `RankMismatch` if `logits` is not rank 2 or 3; `ShapeMismatch` if
/// the target count is not `N`; `IndexOutOfBounds` for a target that
/// is non-integral, negative, or >= `V`.
///
/// # Panics
/// Panics if a candle kernel fails on the validated `[N, V]` logits.
#[allow(clippy::cast_precision_loss)] // `n` is a row count; the divide is exact.
pub fn cross_entropy(logits: &DenseArray, targets: &DenseArray) -> Result<DenseArray, ArrayError> {
    let (n, v) = shape_nv(logits)?;
    let idx = validate_targets(targets, n, v)?;
    let lse = row_logsumexp(logits.data(), n, v);
    let data = logits.data();
    let total: f64 = idx
        .iter()
        .enumerate()
        .map(|(i, &ti)| lse[i] - data[i * v + ti])
        .sum();
    Ok(DenseArray::from_scalar(total / n as f64))
}

/// Resolve `logits` to `[N, V]`, flattening a `[B, T, V]` batch.
///
/// # Errors
/// `RankMismatch` if `logits` is not rank 2 or 3.
fn shape_nv(logits: &DenseArray) -> Result<(usize, usize), ArrayError> {
    match logits.shape().dims() {
        [n, v] => Ok((*n, *v)),
        [b, t, v] => Ok((b * t, *v)),
        d => Err(ArrayError::RankMismatch {
            expected: 2,
            got: d.len(),
        }),
    }
}

/// Validate `targets` is `n` integral indices in `[0, v)`.
///
/// # Errors
/// `ShapeMismatch` if the count is not `n`; `IndexOutOfBounds` for a
/// non-integral, negative, or out-of-range target.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // checked first.
fn validate_targets(targets: &DenseArray, n: usize, v: usize) -> Result<Vec<usize>, ArrayError> {
    if targets.elem_count() != n {
        return Err(ArrayError::ShapeMismatch {
            source: n,
            target: targets.elem_count(),
        });
    }
    let in_range = |t: f64| t >= 0.0 && t.fract() == 0.0 && (t as usize) < v;
    targets
        .data()
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            in_range(t)
                .then_some(t as usize)
                .ok_or(ArrayError::IndexOutOfBounds {
                    axis: 0,
                    index: i,
                    size: v,
                })
        })
        .collect()
}

/// Per-row log-sum-exp of `[n, v]` logits on the GPU, read back as a
/// length-`n` f64 vec for the CPU gather.
fn row_logsumexp(logits: &[f64], n: usize, v: usize) -> Vec<f64> {
    let t = dense_to_cuda(logits, &[n, v]);
    let m = t.max_keepdim(1).expect("cuda row max");
    let sumexp = t
        .broadcast_sub(&m)
        .expect("shift by row max")
        .exp()
        .expect("cuda exp")
        .sum(1)
        .expect("cuda row sum");
    let lse = (sumexp.log().expect("cuda log") + m.squeeze(1).expect("squeeze")).expect("lse add");
    cuda_to_dense_data(&lse)
}
