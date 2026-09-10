//! General element-wise broadcasting: NumPy / APL trailing-axis rules.
//! Shapes align from the RIGHT; a missing leading axis or an axis of
//! extent 1 broadcasts against the other operand. This subsumes scalar
//! and single-element broadcast (a scalar is all-1s, and two single
//! elements take the higher-rank shape) AND enables rank broadcasting --
//! e.g. a rank-3 kernel `[C, kh, kw]` meets rank-5 patches
//! `[oy, ox, C, kh, kw]` so `kernel * windows(...)` needs no reshape.

use mlpl_array::{ArrayError, DenseArray, Shape};

/// Element-wise apply with full trailing-axis broadcasting -> `(data,
/// shape)`; labels are handled by the caller. Equal shapes take a direct
/// zip; otherwise the broadcast output shape is computed and each output
/// element gathered from its (possibly broadcast) source indices.
pub(crate) fn broadcast_apply(
    a: &DenseArray,
    b: &DenseArray,
    op: fn(f64, f64) -> f64,
) -> Result<(Vec<f64>, Shape), ArrayError> {
    if a.shape() == b.shape() {
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(x, y)| op(*x, *y))
            .collect();
        return Ok((data, a.shape().clone()));
    }
    let out = broadcast_shape(a.shape().dims(), b.shape().dims())?;
    let data = broadcast_gather(a, b, &out, op);
    Ok((data, Shape::new(out)))
}

/// The broadcast output shape (per-axis max, right-aligned), or
/// `ShapeMismatch` when two axes differ and neither is 1.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let r = a.len().max(b.len());
    let dim = |d: &[usize], i: usize| {
        let off = r - d.len();
        if i < off { 1 } else { d[i - off] }
    };
    (0..r)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (ad, bd) if ad == bd || bd == 1 => Ok(ad),
            (1, bd) => Ok(bd),
            (ad, bd) => Err(ArrayError::ShapeMismatch {
                source: ad,
                target: bd,
            }),
        })
        .collect()
}

/// Row-major strides of `dims` mapped onto the `out`-rank axes: a
/// prepended axis or an extent-1 axis gets stride 0 (it broadcasts).
fn broadcast_strides(dims: &[usize], out: &[usize]) -> Vec<usize> {
    let off = out.len() - dims.len();
    let mut row = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        row[k] = row[k + 1] * dims[k + 1];
    }
    (0..out.len())
        .map(|k| {
            if k < off || dims[k - off] == 1 {
                0
            } else {
                row[k - off]
            }
        })
        .collect()
}

/// Fill the output by walking each flat index, decomposing it into the
/// output multi-index, and reading each operand through its broadcast
/// strides (stride 0 axes stay pinned at index 0).
fn broadcast_gather(
    a: &DenseArray,
    b: &DenseArray,
    out: &[usize],
    op: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let a_str = broadcast_strides(a.shape().dims(), out);
    let b_str = broadcast_strides(b.shape().dims(), out);
    let mut out_str = vec![1usize; out.len()];
    for k in (0..out.len().saturating_sub(1)).rev() {
        out_str[k] = out_str[k + 1] * out[k + 1];
    }
    let total: usize = out.iter().product::<usize>().max(1);
    (0..total)
        .map(|o| {
            let (mut ai, mut bi, mut rem) = (0usize, 0usize, o);
            for k in 0..out.len() {
                let idx = rem / out_str[k];
                rem %= out_str[k];
                ai += idx * a_str[k];
                bi += idx * b_str[k];
            }
            op(a.data()[ai], b.data()[bi])
        })
        .collect()
}
