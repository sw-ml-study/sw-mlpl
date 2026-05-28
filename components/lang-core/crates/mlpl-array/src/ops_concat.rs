//! Saga 32 step 005: axis-reorganization ops (patchify,
//! stack/concat, take) extracted from `ops.rs` so the
//! parent module stays under the sw-checklist file-LOC budget.
//!
//! Pure data motion -- no new behaviour.

use crate::dense::DenseArray;
use crate::error::ArrayError;
use crate::shape::Shape;

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
