//! Array operations: reshape, transpose, element-wise arithmetic.

use crate::dense::DenseArray;
use crate::error::ArrayError;
use crate::shape::Shape;

impl DenseArray {
    /// Reshape to a new shape, preserving element order.
    ///
    /// Succeeds only when the new shape has the same element count.
    pub fn reshape(&self, new_shape: Shape) -> Result<DenseArray, ArrayError> {
        let source = self.elem_count();
        let target = new_shape.elem_count();
        if source != target {
            return Err(ArrayError::ShapeMismatch { source, target });
        }
        Ok(DenseArray {
            shape: new_shape,
            data: self.data.clone(),
            labels: None,
        })
    }

    /// Transpose: reverse axis order and reorder data to row-major.
    ///
    /// - Scalar/vector: returns a clone (identity).
    /// - Matrix and higher: reverses dims and physically reorders data.
    #[must_use]
    pub fn transpose(&self) -> DenseArray {
        let dims = self.shape().dims();
        if dims.len() <= 1 {
            return self.clone();
        }

        let new_dims: Vec<usize> = dims.iter().rev().copied().collect();
        let new_shape = Shape::new(new_dims);
        let n = self.elem_count();
        let mut new_data = vec![0.0; n];

        let rank = dims.len();
        let old_strides = compute_strides(dims);
        let new_strides = compute_strides(new_shape.dims());

        for flat in 0..n {
            let mut remainder = flat;
            let mut new_flat = 0;
            for axis in 0..rank {
                let idx = remainder / old_strides[axis];
                remainder %= old_strides[axis];
                new_flat += idx * new_strides[rank - 1 - axis];
            }
            new_data[new_flat] = self.data[flat];
        }

        let labels = self
            .labels
            .as_ref()
            .map(|lbls| lbls.iter().rev().cloned().collect());

        DenseArray {
            shape: new_shape,
            data: new_data,
            labels,
        }
    }
}

impl DenseArray {
    /// Apply a binary operation element-wise with scalar broadcasting.
    ///
    /// - Same shape: element-wise.
    /// - One scalar: broadcast to the other's shape.
    /// - Otherwise: ShapeMismatch error.
    ///
    /// Label propagation (Saga 11.5 Phase 3): scalar operands contribute
    /// no labels, so the non-scalar side's labels win unconditionally.
    /// For same-shape operands: if either side is unlabeled, the
    /// labeled side's labels carry through; if both are labeled, the
    /// label vectors must match or `LabelMismatch` is raised.
    pub fn apply_binop(
        &self,
        other: &DenseArray,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let result_labels = merge_labels(self, other)?;
        if self.shape() == other.shape() {
            let data: Vec<f64> = self
                .data()
                .iter()
                .zip(other.data().iter())
                .map(|(a, b)| op(*a, *b))
                .collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        // Scalar broadcast
        if self.rank() == 0 {
            let s = self.data()[0];
            let data: Vec<f64> = other.data().iter().map(|b| op(s, *b)).collect();
            return Ok(DenseArray {
                shape: other.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        if other.rank() == 0 {
            let s = other.data()[0];
            let data: Vec<f64> = self.data().iter().map(|a| op(*a, s)).collect();
            return Ok(DenseArray {
                shape: self.shape.clone(),
                data,
                labels: result_labels,
            });
        }
        Err(ArrayError::ShapeMismatch {
            source: self.elem_count(),
            target: other.elem_count(),
        })
    }
}

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
}

impl DenseArray {
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

impl DenseArray {
    /// Reduce along an axis using the given binary operation.
    ///
    /// Removes the specified axis from the shape. For example,
    /// a [2,3] array reduced along axis 0 produces a [3] result.
    pub fn reduce_axis(
        &self,
        axis: usize,
        identity: f64,
        op: fn(f64, f64) -> f64,
    ) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        if axis >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis,
                index: axis,
                size: dims.len(),
            });
        }
        let mut result_dims: Vec<usize> = dims.to_vec();
        result_dims.remove(axis);
        let result_shape = Shape::new(result_dims);
        let result_count = result_shape.elem_count();
        let mut result_data = vec![identity; result_count];

        let strides = compute_strides(dims);
        let axis_size = dims[axis];
        let axis_stride = strides[axis];

        for flat in 0..self.elem_count() {
            let result_flat = if axis_stride > 1 {
                let outer = flat / (axis_size * axis_stride);
                let inner = flat % axis_stride;
                outer * axis_stride + inner
            } else {
                flat / axis_size
            };
            result_data[result_flat] = op(result_data[result_flat], self.data[flat]);
        }

        let labels = self.labels.as_ref().map(|lbls| {
            let mut out = lbls.clone();
            out.remove(axis);
            out
        });
        Ok(DenseArray {
            shape: result_shape,
            data: result_data,
            labels,
        })
    }
}

impl DenseArray {
    /// Argmax along an axis. Returns an array with the given axis
    /// removed whose values are the indices (as f64) of the maxima.
    /// Ties go to the first occurrence.
    pub fn argmax_axis(&self, axis: usize) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        if axis >= dims.len() {
            return Err(ArrayError::IndexOutOfBounds {
                axis,
                index: axis,
                size: dims.len(),
            });
        }
        let mut result_dims: Vec<usize> = dims.to_vec();
        result_dims.remove(axis);
        let result_shape = Shape::new(result_dims);
        let result_count = result_shape.elem_count();
        let mut best_val = vec![f64::NEG_INFINITY; result_count];
        let mut best_idx = vec![0.0f64; result_count];

        let strides = compute_strides(dims);
        let axis_size = dims[axis];
        let axis_stride = strides[axis];

        for flat in 0..self.elem_count() {
            let result_flat = if axis_stride > 1 {
                let outer = flat / (axis_size * axis_stride);
                let inner = flat % axis_stride;
                outer * axis_stride + inner
            } else {
                flat / axis_size
            };
            // Recover axis index: (flat / axis_stride) % axis_size.
            let axis_idx = (flat / axis_stride) % axis_size;
            let v = self.data()[flat];
            if v > best_val[result_flat] {
                best_val[result_flat] = v;
                best_idx[result_flat] = axis_idx as f64;
            }
        }

        let mut out = DenseArray::new(result_shape, best_idx)?;
        if let Some(lbls) = self.labels.as_ref() {
            let mut new_lbls = lbls.clone();
            new_lbls.remove(axis);
            out.labels = Some(new_lbls);
        }
        Ok(out)
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

/// Compute the label list for the result of an elementwise op on two
/// arrays. Scalars contribute no labels, so the non-scalar side wins
/// unconditionally when one operand is rank 0. For non-scalar pairs,
/// two unlabeled sides stay unlabeled, a single labeled side carries
/// its labels through, and two labeled sides must agree or
/// `LabelMismatch` is returned. Saga 11.5 Phase 3.
fn merge_labels(a: &DenseArray, b: &DenseArray) -> Result<Option<Vec<Option<String>>>, ArrayError> {
    if a.rank() == 0 {
        return Ok(b.labels.clone());
    }
    if b.rank() == 0 {
        return Ok(a.labels.clone());
    }
    match (&a.labels, &b.labels) {
        (None, None) => Ok(None),
        (Some(l), None) | (None, Some(l)) => Ok(Some(l.clone())),
        (Some(la), Some(lb)) if la == lb => Ok(Some(la.clone())),
        (Some(la), Some(lb)) => Err(ArrayError::LabelMismatch {
            expected: la.clone(),
            actual: lb.clone(),
        }),
    }
}

/// Compute strides for row-major layout.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

impl DenseArray {
    /// Saga 29 step 005: ViT patch embedding rearrangement.
    /// Take a `[B, C, H, W]` image batch and a square patch
    /// size `P` (which must divide both `H` and `W`). Return
    /// `[B, N, P*P*C]` where `N = (H/P) * (W/P)` and each
    /// row of the trailing axis is one patch flattened with
    /// channel-outer order: index `c * P*P + dy * P + dx`.
    ///
    /// Patch traversal order across `N` is row-major:
    /// `n = i * (W/P) + j` for patch row `i` (top-to-bottom)
    /// and column `j` (left-to-right).
    pub fn patchify(&self, patch_size: usize) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        if dims.len() != 4 {
            return Err(ArrayError::ShapeMismatch {
                source: dims.len(),
                target: 4,
            });
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if patch_size == 0 || h % patch_size != 0 || w % patch_size != 0 {
            return Err(ArrayError::ShapeMismatch {
                source: patch_size,
                target: h.min(w),
            });
        }
        let p = patch_size;
        let nh = h / p;
        let nw = w / p;
        let n_patches = nh * nw;
        let patch_len = p * p * c;
        let mut out = vec![0.0; b * n_patches * patch_len];
        let src = self.data();
        for b_i in 0..b {
            for i in 0..nh {
                for j in 0..nw {
                    let n = i * nw + j;
                    for c_i in 0..c {
                        for dy in 0..p {
                            for dx in 0..p {
                                let src_y = i * p + dy;
                                let src_x = j * p + dx;
                                let src_idx = ((b_i * c + c_i) * h + src_y) * w + src_x;
                                let dst_idx =
                                    (b_i * n_patches + n) * patch_len + c_i * p * p + dy * p + dx;
                                out[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
        }
        DenseArray::new(Shape::new(vec![b, n_patches, patch_len]), out)
    }

    /// Concat two arrays along `axis`. Both inputs must agree on
    /// every dim except `axis`, where the sizes add. Any axis in
    /// `[0, rank)` is supported (Saga 30 step 001 lifted the
    /// original `{0, 1}` restriction from Saga 29 step 005).
    /// Labels are taken from `self` -- the `other` operand's
    /// labels are ignored, matching how concat composes inputs
    /// of the same semantic shape.
    pub fn concat(&self, other: &DenseArray, axis: usize) -> Result<DenseArray, ArrayError> {
        let a_dims = self.shape().dims();
        let b_dims = other.shape().dims();
        if a_dims.len() != b_dims.len() {
            return Err(ArrayError::ShapeMismatch {
                source: a_dims.len(),
                target: b_dims.len(),
            });
        }
        if axis >= a_dims.len() {
            return Err(ArrayError::ShapeMismatch {
                source: axis,
                target: a_dims.len(),
            });
        }
        for (k, (&a, &b)) in a_dims.iter().zip(b_dims.iter()).enumerate() {
            if k != axis && a != b {
                return Err(ArrayError::ShapeMismatch {
                    source: a,
                    target: b,
                });
            }
        }
        let mut out_dims = a_dims.to_vec();
        out_dims[axis] = a_dims[axis] + b_dims[axis];
        let mut data = Vec::with_capacity(out_dims.iter().product());
        copy_concat_rows(&mut data, self, other, axis);
        let arr = DenseArray::new(Shape::new(out_dims), data)?;
        if let Some(lbls) = self.labels() {
            arr.with_labels(lbls.to_vec())
        } else {
            Ok(arr)
        }
    }
}

impl DenseArray {
    /// Saga 29 step 013: N-way concatenation along an existing
    /// axis. All inputs must have identical shape. Output shape
    /// matches the input shape with `dims[axis]` multiplied by
    /// `n`. Used by the multi-head attention tape lowering (stack
    /// per-head outputs along the column axis) and the rank-3
    /// batched attention path (stack per-batch outputs along the
    /// batch axis) to replace the previous O(N^2) binary-concat
    /// chain with a single O(N) op.
    pub fn stack(arrays: &[&DenseArray], axis: usize) -> Result<DenseArray, ArrayError> {
        if arrays.is_empty() {
            return Err(ArrayError::ShapeMismatch {
                source: 0,
                target: 1,
            });
        }
        let first = arrays[0];
        let dims = first.shape().dims();
        if axis >= dims.len() {
            return Err(ArrayError::ShapeMismatch {
                source: axis,
                target: dims.len(),
            });
        }
        for a in arrays.iter().skip(1) {
            if a.shape().dims() != dims {
                return Err(ArrayError::ShapeMismatch {
                    source: a.shape().dims().len(),
                    target: dims.len(),
                });
            }
        }
        let n = arrays.len();
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
        let parent_stride = dims[axis] * inner;
        let mut out_dims = dims.to_vec();
        out_dims[axis] = n * dims[axis];
        let mut data: Vec<f64> = Vec::with_capacity(outer * n * parent_stride);
        for o in 0..outer {
            for arr in arrays {
                let start = o * parent_stride;
                data.extend_from_slice(&arr.data()[start..start + parent_stride]);
            }
        }
        let arr = DenseArray::new(Shape::new(out_dims), data)?;
        if let Some(lbls) = first.labels() {
            arr.with_labels(lbls.to_vec())
        } else {
            Ok(arr)
        }
    }
}

/// Helper for `DenseArray::concat`: copy chunks from `a` then
/// `b` along `axis`. Generalized for any axis (Saga 30 step 001)
/// by walking the outer dims (`dims[..axis]`) and, for each
/// outer position, copying that position's slab of `a`
/// (`a_dims[axis] * inner` elements) then `b`
/// (`b_dims[axis] * inner` elements). `inner` is the product of
/// `dims[axis+1..]`, defaulting to 1 when `axis` is the last
/// dimension. Reduces to the original axis-0 / axis-1 fast
/// paths when those axes are picked.
fn copy_concat_rows(out: &mut Vec<f64>, a: &DenseArray, b: &DenseArray, axis: usize) {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let outer: usize = a_dims[..axis].iter().product();
    let inner: usize = a_dims[axis + 1..].iter().product::<usize>().max(1);
    let a_slab = a_dims[axis] * inner;
    let b_slab = b_dims[axis] * inner;
    let a_data = a.data();
    let b_data = b.data();
    for o in 0..outer {
        let a_start = o * a_slab;
        let b_start = o * b_slab;
        out.extend_from_slice(&a_data[a_start..a_start + a_slab]);
        out.extend_from_slice(&b_data[b_start..b_start + b_slab]);
    }
}

impl DenseArray {
    /// Saga 29 step 007: drop one axis at a single integer
    /// index. `axis` must be in `[0, rank)` and `idx` must be
    /// in `[0, dims[axis])`. Per-axis labels propagate except
    /// for the dropped axis. Used by ViT to pull one image out
    /// of a `[B, C, H, W]` batch (`take(X, 0, i)`) and to read
    /// the CLS row of a `[B, T, d]` post-attention activation
    /// (`take(seq, 1, 0)`).
    pub fn take(&self, axis: usize, idx: usize) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        if axis >= dims.len() {
            return Err(ArrayError::ShapeMismatch {
                source: axis,
                target: dims.len(),
            });
        }
        if idx >= dims[axis] {
            return Err(ArrayError::ShapeMismatch {
                source: idx,
                target: dims[axis],
            });
        }
        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
        let mut out = Vec::with_capacity(outer * inner);
        let data = self.data();
        for o in 0..outer {
            let start = (o * axis_size + idx) * inner;
            out.extend_from_slice(&data[start..start + inner]);
        }
        let mut new_dims = Vec::with_capacity(dims.len().saturating_sub(1));
        for (k, &d) in dims.iter().enumerate() {
            if k != axis {
                new_dims.push(d);
            }
        }
        let arr = DenseArray::new(Shape::new(new_dims), out)?;
        if let Some(lbls) = self.labels() {
            let kept: Vec<Option<String>> = lbls
                .iter()
                .enumerate()
                .filter_map(|(k, l)| if k == axis { None } else { Some(l.clone()) })
                .collect();
            arr.with_labels(kept)
        } else {
            Ok(arr)
        }
    }
}
