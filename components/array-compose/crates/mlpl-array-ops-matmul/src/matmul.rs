use mlpl_array::{ArrayError, DenseArray, Shape};
use rayon::prelude::*;

use crate::labels::matmul_labels;

/// Parallelize the row loop only when there are enough rows to amortize
/// rayon's fan-out cost; tiny matmuls (the common case in interactive
/// REPL ops) stay sequential to avoid thread-pool overhead.
const PAR_ROW_THRESHOLD: usize = 64;

/// Matrix-multiplication extension for `DenseArray`. Bring into
/// scope with `use mlpl_array_ops_matmul::prelude::*;` to call
/// `a.matmul(&b)`.
pub trait MatmulExt {
    /// Matrix multiplication:
    /// - `[m, k] * [k, n] -> [m, n]`
    /// - `[m, k] * [k]    -> [m]`  (matrix-vector product)
    fn matmul(&self, other: &DenseArray) -> Result<DenseArray, ArrayError>;
}

impl MatmulExt for DenseArray {
    fn matmul(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        let (m, k) = match self.shape().dims() {
            [m, k] => (*m, *k),
            _ => {
                return Err(ArrayError::RankMismatch {
                    expected: 2,
                    got: self.rank(),
                });
            }
        };
        let labels = matmul_labels(self, other)?;
        let (data, out_shape) = match other.shape().dims() {
            [k2, n] if *k2 == k => (mat_mat_data(self, other, k, *n), Shape::new(vec![m, *n])),
            [k2] if *k2 == k => (mat_vec_data(self, other, k), Shape::vector(m)),
            _ => {
                return Err(ArrayError::ShapeMismatch {
                    source: k,
                    target: other.shape().dims().first().copied().unwrap_or(0),
                });
            }
        };
        let arr = DenseArray::new(out_shape, data)?;
        match labels {
            Some(l) => arr.with_labels(l),
            None => Ok(arr),
        }
    }
}

/// One output row: `row . B`, column by column. `b` is row-major `[k, n]`.
fn row_times_mat(row: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            row.iter()
                .zip(b.chunks(n).map(|col| col[j]))
                .map(|(x, y)| x * y)
                .sum::<f64>()
        })
        .collect()
}

/// `[m, k] * [k, n] -> [m, n]`, row-major. The independent output rows
/// fan out across cores via rayon (ordered collect == identical result
/// to the sequential path) once there are enough rows to be worth it.
fn mat_mat_data(a: &DenseArray, b: &DenseArray, k: usize, n: usize) -> Vec<f64> {
    let (a_data, b_data) = (a.data(), b.data());
    let rows = a_data.len() / k.max(1);
    if rows >= PAR_ROW_THRESHOLD {
        a_data
            .par_chunks(k)
            .flat_map_iter(|row| row_times_mat(row, b_data, n))
            .collect()
    } else {
        a_data
            .chunks(k)
            .flat_map(|row| row_times_mat(row, b_data, n))
            .collect()
    }
}

fn mat_vec_data(a: &DenseArray, b: &DenseArray, k: usize) -> Vec<f64> {
    a.data()
        .chunks(k)
        .map(|row| row.iter().zip(b.data().iter()).map(|(x, y)| x * y).sum())
        .collect()
}
