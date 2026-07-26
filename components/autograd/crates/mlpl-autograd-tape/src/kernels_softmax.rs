//! Softmax forward/backward kernels + the gradient accumulator.
//! Split from `ops.rs` (tech-debt spike step 002).

use mlpl_array::DenseArray;
use mlpl_array_ops_element::prelude::*;

/// Numerically-stable softmax along `axis`, supports rank-1 and rank-2.
#[must_use]
pub fn softmax_forward(x: &DenseArray, axis: usize) -> DenseArray {
    let row = |r: &[f64]| -> Vec<f64> {
        let m = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = r.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / s).collect()
    };
    let dims = x.shape().dims().to_vec();
    match dims.len() {
        0 => x.clone(),
        1 => DenseArray::new(x.shape().clone(), row(x.data())).expect("shape"),
        2 => {
            let (rows, cols) = (dims[0], dims[1]);
            let mut out = vec![0.0; rows * cols];
            if axis == 1 {
                for r in 0..rows {
                    let sm = row(&x.data()[r * cols..(r + 1) * cols]);
                    out[r * cols..(r + 1) * cols].copy_from_slice(&sm);
                }
            } else {
                for c in 0..cols {
                    let col: Vec<f64> = (0..rows).map(|r| x.data()[r * cols + c]).collect();
                    let sm = row(&col);
                    for (r, v) in sm.iter().enumerate() {
                        out[r * cols + c] = *v;
                    }
                }
            }
            DenseArray::new(x.shape().clone(), out).expect("shape")
        }
        _ => panic!("softmax supports rank <= 2"),
    }
}

/// Softmax backward given forward output `y` and upstream grad `g`.
///
/// For each row: `dx_i = y_i * (g_i - sum_j(g_j * y_j))`.
#[must_use]
pub fn softmax_backward(y: &DenseArray, upstream: &DenseArray, axis: usize) -> DenseArray {
    let row_bw = |yr: &[f64], gr: &[f64]| -> Vec<f64> {
        let dot: f64 = yr.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
        yr.iter()
            .zip(gr.iter())
            .map(|(yi, gi)| yi * (gi - dot))
            .collect()
    };
    let dims = y.shape().dims().to_vec();
    match dims.len() {
        0 => DenseArray::from_scalar(0.0),
        1 => DenseArray::new(y.shape().clone(), row_bw(y.data(), upstream.data())).expect("shape"),
        2 => softmax_backward_2d(y, upstream, axis, (dims[0], dims[1])),
        _ => panic!("softmax supports rank <= 2"),
    }
}

/// Rank-2 softmax backward: per-row when `axis == 1`, per-column
/// otherwise.
fn softmax_backward_2d(
    y: &DenseArray,
    upstream: &DenseArray,
    axis: usize,
    (rows, cols): (usize, usize),
) -> DenseArray {
    let row_bw = |yr: &[f64], gr: &[f64]| -> Vec<f64> {
        let dot: f64 = yr.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
        yr.iter()
            .zip(gr.iter())
            .map(|(yi, gi)| yi * (gi - dot))
            .collect()
    };
    let mut out = vec![0.0; rows * cols];
    if axis == 1 {
        for r in 0..rows {
            let dx = row_bw(
                &y.data()[r * cols..(r + 1) * cols],
                &upstream.data()[r * cols..(r + 1) * cols],
            );
            out[r * cols..(r + 1) * cols].copy_from_slice(&dx);
        }
    } else {
        for c in 0..cols {
            let yc: Vec<f64> = (0..rows).map(|r| y.data()[r * cols + c]).collect();
            let gc: Vec<f64> = (0..rows).map(|r| upstream.data()[r * cols + c]).collect();
            let dx = row_bw(&yc, &gc);
            for (r, v) in dx.iter().enumerate() {
                out[r * cols + c] = *v;
            }
        }
    }
    DenseArray::new(y.shape().clone(), out).expect("shape")
}

/// Accumulate `incoming` into `slot`: add if already present, else set.
pub fn accumulate(slot: &mut Option<DenseArray>, incoming: DenseArray) {
    match slot {
        None => *slot = Some(incoming),
        Some(existing) => {
            let sum = existing
                .apply_binop(&incoming, |a, b| a + b)
                .expect("matching shapes during accumulation");
            *existing = sum;
        }
    }
}
