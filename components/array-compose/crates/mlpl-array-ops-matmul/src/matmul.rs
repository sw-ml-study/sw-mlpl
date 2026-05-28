use mlpl_array::{ArrayError, DenseArray, Shape};

use crate::labels::matmul_labels;

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

fn mat_mat_data(a: &DenseArray, b: &DenseArray, k: usize, n: usize) -> Vec<f64> {
    a.data()
        .chunks(k)
        .flat_map(|row| {
            (0..n).map(move |j| {
                row.iter()
                    .zip(b.data().chunks(n).map(|col| col[j]))
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
            })
        })
        .collect()
}

fn mat_vec_data(a: &DenseArray, b: &DenseArray, k: usize) -> Vec<f64> {
    a.data()
        .chunks(k)
        .map(|row| row.iter().zip(b.data().iter()).map(|(x, y)| x * y).sum())
        .collect()
}
