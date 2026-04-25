//! Reductions, normalisation, and loss primitives (Saga 14 step 003).
//!
//! Each function mirrors the runtime-target signature that the
//! `mlpl-mlx-rt` sibling exposes, so compiled MLPL code can swap
//! runtimes without source changes. Semantics match the
//! interpreter's `mlpl-runtime` builtins by construction:
//! `softmax` and `log_softmax` use the same max-subtraction trick
//! for numerical stability, and `cross_entropy` is the matching
//! fused log-softmax + NLL with a scalar mean output.
//!
//! Label propagation follows Saga 11.5: a reduced axis drops its
//! label; `softmax` and `log_softmax` preserve labels (output
//! shape is unchanged); `cross_entropy` returns an unlabeled
//! scalar.

use mlpl_array::{ArrayError, DenseArray, Shape};

/// Product reduction. `axis = None` collapses to a scalar over the
/// flat data; `axis = Some(ax)` removes one dimension and returns a
/// rank-(r-1) array.
pub fn reduce_mul(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    match axis {
        Some(ax) => a.reduce_axis(ax, 1.0, |x, y| x * y),
        None => Ok(DenseArray::from_scalar(a.data().iter().copied().product())),
    }
}

/// Mean reduction. Flat = sum / elem_count (with elem_count clamped
/// to 1 for the empty case so the result is finite); axis-aware =
/// per-axis sum divided by the reduced axis size.
pub fn mean(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    if let Some(ax) = axis {
        let dims = a.shape().dims();
        if ax >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis: ax,
                index: ax,
                size: dims.len(),
            });
        }
        let n = dims[ax].max(1) as f64;
        let s = a.reduce_axis(ax, 0.0, |x, y| x + y)?;
        let scaled: Vec<f64> = s.data().iter().map(|v| v / n).collect();
        return DenseArray::new(s.shape().clone(), scaled).and_then(|arr| match s.labels() {
            Some(lbls) => arr.with_labels(lbls.to_vec()),
            None => Ok(arr),
        });
    }
    let n = a.elem_count().max(1) as f64;
    let s: f64 = a.data().iter().sum();
    Ok(DenseArray::from_scalar(s / n))
}

/// Index of the maximum value. Flat returns a rank-0 scalar holding
/// the flat index of the global max; axis-aware delegates to
/// `DenseArray::argmax_axis` (drops the chosen axis from the result).
/// Ties go to the first occurrence on both paths.
pub fn argmax(a: &DenseArray, axis: Option<usize>) -> Result<DenseArray, ArrayError> {
    if let Some(ax) = axis {
        return a.argmax_axis(ax);
    }
    let data = a.data();
    if data.is_empty() {
        return Err(ArrayError::EmptyArray);
    }
    let mut best_idx = 0usize;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(DenseArray::from_scalar(best_idx as f64))
}

/// `softmax(a, axis)`: exp-and-normalise along `axis` with the
/// max-subtraction trick for numerical stability. Output shape and
/// labels match the input.
pub fn softmax(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
    let dims = a.shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(ArrayError::IndexOutOfBounds {
            axis,
            index: axis,
            size: dims.len(),
        });
    }
    let axis_size = dims[axis];
    let axis_stride: usize = dims.iter().skip(axis + 1).product();
    let group_count = (a.elem_count() / axis_size).max(1);
    let group_of = |flat: usize| {
        if axis_stride > 1 {
            (flat / (axis_size * axis_stride)) * axis_stride + (flat % axis_stride)
        } else {
            flat / axis_size
        }
    };
    let mut maxv = vec![f64::NEG_INFINITY; group_count];
    for (flat, &v) in a.data().iter().enumerate() {
        let g = group_of(flat);
        if v > maxv[g] {
            maxv[g] = v;
        }
    }
    let mut out = vec![0.0f64; a.elem_count()];
    let mut sums = vec![0.0f64; group_count];
    for (flat, slot) in out.iter_mut().enumerate() {
        let g = group_of(flat);
        let e = (a.data()[flat] - maxv[g]).exp();
        *slot = e;
        sums[g] += e;
    }
    for (flat, slot) in out.iter_mut().enumerate() {
        *slot /= sums[group_of(flat)];
    }
    let result = DenseArray::new(Shape::new(dims), out)?;
    match a.labels() {
        Some(lbls) => result.with_labels(lbls.to_vec()),
        None => Ok(result),
    }
}

/// `log_softmax(a, axis)`: same shape as input, equal to
/// `a - logsumexp(a, axis)` along `axis`. Computed via the same
/// max-subtraction LSE trick as `softmax` so the two stay
/// numerically consistent.
pub fn log_softmax(a: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
    let dims = a.shape().dims().to_vec();
    if axis >= dims.len() {
        return Err(ArrayError::IndexOutOfBounds {
            axis,
            index: axis,
            size: dims.len(),
        });
    }
    let axis_size = dims[axis];
    let axis_stride: usize = dims.iter().skip(axis + 1).product();
    let group_count = (a.elem_count() / axis_size).max(1);
    let group_of = |flat: usize| {
        if axis_stride > 1 {
            (flat / (axis_size * axis_stride)) * axis_stride + (flat % axis_stride)
        } else {
            flat / axis_size
        }
    };
    let mut maxv = vec![f64::NEG_INFINITY; group_count];
    for (flat, &v) in a.data().iter().enumerate() {
        let g = group_of(flat);
        if v > maxv[g] {
            maxv[g] = v;
        }
    }
    let mut sums = vec![0.0f64; group_count];
    for (flat, &v) in a.data().iter().enumerate() {
        let g = group_of(flat);
        sums[g] += (v - maxv[g]).exp();
    }
    let lse: Vec<f64> = sums
        .iter()
        .zip(maxv.iter())
        .map(|(s, m)| m + s.ln())
        .collect();
    let mut out = vec![0.0f64; a.elem_count()];
    for (flat, slot) in out.iter_mut().enumerate() {
        *slot = a.data()[flat] - lse[group_of(flat)];
    }
    let result = DenseArray::new(Shape::new(dims), out)?;
    match a.labels() {
        Some(lbls) => result.with_labels(lbls.to_vec()),
        None => Ok(result),
    }
}

/// `cross_entropy(logits, targets)`: fused log-softmax + NLL,
/// returned as a scalar mean over rows. `logits` must be rank 2
/// (`[N, V]`) or rank 3 (`[B, T, V]`); `targets` holds non-negative
/// integer values whose count equals the row count (`N` or `B*T`).
/// Always returns an unlabeled scalar, matching the CPU
/// `mlpl_runtime::call_builtin("cross_entropy", _)` behaviour.
pub fn cross_entropy(logits: &DenseArray, targets: &DenseArray) -> Result<DenseArray, ArrayError> {
    let dims = logits.shape().dims();
    let (n, v) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        r => {
            return Err(ArrayError::RankMismatch {
                expected: 2,
                got: r,
            });
        }
    };
    if targets.elem_count() != n {
        return Err(ArrayError::ShapeMismatch {
            source: n,
            target: targets.elem_count(),
        });
    }
    let data = logits.data();
    let mut total = 0.0;
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 {
            return Err(ArrayError::IndexOutOfBounds {
                axis: 0,
                index: i,
                size: v,
            });
        }
        let ti = t as usize;
        if ti >= v {
            return Err(ArrayError::IndexOutOfBounds {
                axis: 0,
                index: ti,
                size: v,
            });
        }
        let row = &data[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - row[ti];
    }
    Ok(DenseArray::from_scalar(total / n as f64))
}
