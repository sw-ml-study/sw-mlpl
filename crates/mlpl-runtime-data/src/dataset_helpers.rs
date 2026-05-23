//! Saga 32 step 006: pure permutation + gather helpers
//! extracted from `dataset_builtins.rs` to keep the
//! orchestrator under the sw-checklist function-count budget.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::{RuntimeError, Xorshift64};

/// Fisher-Yates shuffle producing a permutation `[0, n)`
/// driven by a seeded `Xorshift64`. Deterministic across
/// platforms because the PRNG state is explicit.
pub(crate) fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng = Xorshift64::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// Gather rows of `x` at the given `indices`. Preserves any
/// per-axis labels except axis 0 (which is the gathered axis;
/// per-row labels would not survive a reordering).
pub(crate) fn gather_rows(x: &DenseArray, indices: &[usize]) -> Result<DenseArray, RuntimeError> {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(RuntimeError::InvalidArgument {
            func: "gather_rows".into(),
            reason: "rank >= 1 required".into(),
        });
    }
    let row_stride: usize = dims[1..].iter().product::<usize>().max(1);
    let mut data = Vec::with_capacity(indices.len() * row_stride);
    let src = x.data();
    for &i in indices {
        data.extend_from_slice(&src[i * row_stride..(i + 1) * row_stride]);
    }
    let mut out_dims = vec![indices.len()];
    out_dims.extend_from_slice(&dims[1..]);
    let mut out = DenseArray::new(Shape::new(out_dims), data)?;
    if let Some(src_labels) = x.labels() {
        let mut labels = vec![None];
        labels.extend_from_slice(&src_labels[1..]);
        out = out.with_labels(labels)?;
    }
    Ok(out)
}
