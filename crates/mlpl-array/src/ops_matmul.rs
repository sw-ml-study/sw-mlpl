//! Saga 33 step 005: dot-product + matmul extracted from `ops.rs`,
//! plus the `matmul_labels` helper for Saga 11.5 label propagation
//! across the contraction axis.

use crate::dense::DenseArray;
use crate::error::ArrayError;
use crate::shape::Shape;

impl DenseArray {
    /// Dot product of two rank-1 vectors.
    ///
    /// Both must be vectors of the same length. Returns a scalar.
    pub fn dot(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(ArrayError::RankMismatch {
                expected: 1,
                got: if self.rank() != 1 {
                    self.rank()
                } else {
                    other.rank()
                },
            });
        }
        if self.elem_count() != other.elem_count() {
            return Err(ArrayError::ShapeMismatch {
                source: self.elem_count(),
                target: other.elem_count(),
            });
        }
        let sum: f64 = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(DenseArray::from_scalar(sum))
    }

    /// Matrix multiplication.
    ///
    /// - [m, k] * [k, n] -> [m, n]
    /// - [m, k] * [k] -> [m] (matrix-vector product)
    ///
    /// Label propagation (Saga 11.5 Phase 3): the contraction axis
    /// (`self`'s last dim against `other`'s first dim) must agree when
    /// both sides carry explicit labels. Result labels are the
    /// non-contracted dims: `[self.labels[0], other.labels[1]]` for
    /// matrix-matrix, `[self.labels[0]]` for matrix-vector. Unlabeled
    /// sides contribute `None` positions.
    pub fn matmul(&self, other: &DenseArray) -> Result<DenseArray, ArrayError> {
        let (m, k) = match self.shape().dims() {
            [m, k] => (*m, *k),
            _ => {
                return Err(ArrayError::RankMismatch {
                    expected: 2,
                    got: self.rank(),
                });
            }
        };
        let result_labels = matmul_labels(self, other)?;
        match other.shape().dims() {
            [k2, n] if *k2 == k => {
                let n = *n;
                let data: Vec<f64> = self
                    .data
                    .chunks(k)
                    .flat_map(|row| {
                        (0..n).map(move |j| {
                            row.iter()
                                .zip(other.data.chunks(n).map(|col| col[j]))
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                        })
                    })
                    .collect();
                Ok(DenseArray {
                    shape: Shape::new(vec![m, n]),
                    data,
                    labels: result_labels,
                })
            }
            [k2] if *k2 == k => {
                let data: Vec<f64> = self
                    .data
                    .chunks(k)
                    .map(|row| row.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
                    .collect();
                Ok(DenseArray {
                    shape: Shape::vector(m),
                    data,
                    labels: result_labels,
                })
            }
            _ => Err(ArrayError::ShapeMismatch {
                source: k,
                target: other.shape().dims().first().copied().unwrap_or(0),
            }),
        }
    }
}

/// Compute the result labels for a `matmul(a, b)`. The contraction
/// axis is `a`'s last dim vs `b`'s first dim; if both sides name it
/// and the names differ, raise `LabelMismatch`. Output labels are the
/// non-contracted dims. An unlabeled side contributes `None` at its
/// position, preserving partial labeling. If neither side is labeled,
/// the result is fully unlabeled. Saga 11.5 Phase 3.
fn matmul_labels(
    a: &DenseArray,
    b: &DenseArray,
) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.labels.is_none() && b.labels.is_none() {
        return Ok(None);
    }
    let default_b = vec![None; b.rank()];
    let al = a.labels.as_ref().map_or(&[None, None][..], Vec::as_slice);
    let bl = b
        .labels
        .as_ref()
        .map_or(default_b.as_slice(), Vec::as_slice);
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
