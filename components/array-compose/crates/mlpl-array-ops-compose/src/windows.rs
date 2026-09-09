//! `windows` -- stride-1 "valid" sliding-window extraction over an
//! array's trailing axes. This is the primitive underlying a moving
//! average (1-D) and a convolution patch stack (2-D): the paper's
//! `x_{x+u, y+v}` becomes an array, so a convolution is
//! `reduce(:add, kernel * windows(img, [kh, kw]))`.
//!
//! Output axis order is `[output-position axes, then the non-windowed
//! (leading) axes, then the window-size axes]`. Each windowed axis of
//! extent `d` with a window of size `w` (stride `s`) becomes a leading
//! output-POSITION axis of extent `(d - w) / s + 1`; the non-windowed
//! axes follow; the window sizes trail:
//! `[H, W].windows([kh, kw])` -> `[out_y, out_x, kh, kw]` and
//! `[C, H, W].windows([kh, kw])` -> `[out_y, out_x, C, kh, kw]`.
//!
//! Positions lead so you index the output grid naturally, and -- the
//! load-bearing choice for convolution -- the non-windowed axes sit
//! next to the window axes as a trailing `[C, kh, kw]` block. A kernel
//! `[C, kh, kw]` therefore aligns by trailing position under NumPy-style
//! rank broadcasting, so `reduce(:add, kernel * windows(img, [kh, kw]))`
//! needs no transpose to line the receptive field up with the weights.

use mlpl_array::{ArrayError, DenseArray, Shape};

/// Sliding-window extraction extension for `DenseArray`.
pub trait WindowsExt {
    /// Every window of the given `sizes` over the last `sizes.len()`
    /// axes, advancing by `strides` (one stride per windowed axis;
    /// stride 1 is dense/overlapping). See the module docs for the
    /// output shape.
    fn windows(&self, sizes: &[usize], strides: &[usize]) -> Result<DenseArray, ArrayError>;
}

impl WindowsExt for DenseArray {
    fn windows(&self, sizes: &[usize], strides: &[usize]) -> Result<DenseArray, ArrayError> {
        let dims = self.shape().dims();
        validate(dims, sizes, strides)?;
        let out_dims = out_dims(dims, sizes, strides);
        let data = gather(self.data(), dims, sizes, strides, &out_dims);
        let arr = DenseArray::new(Shape::new(out_dims), data)?;
        // Position axes inherit the windowed axis labels, the non-windowed
        // axes keep theirs, and the window axes are unlabeled (a demo can
        // relabel them, e.g. kernel_y/kernel_x, for a readable conv).
        match self.labels() {
            Some(lbls) => {
                let (r, k) = (dims.len(), sizes.len());
                let mut out = lbls[r - k..].to_vec(); // positions <- windowed labels
                out.extend_from_slice(&lbls[..r - k]); // non-windowed axes
                out.resize(r + k, None); // window axes unlabeled
                arr.with_labels(out)
            }
            None => Ok(arr),
        }
    }
}

/// `sizes` must window 1..=rank trailing axes, each window in `1..=extent`;
/// `strides` must match `sizes` in length, each stride >= 1.
fn validate(dims: &[usize], sizes: &[usize], strides: &[usize]) -> Result<(), ArrayError> {
    let (r, k) = (dims.len(), sizes.len());
    if k == 0 || k > r || strides.len() != k {
        return Err(ArrayError::ShapeMismatch {
            source: k,
            target: r,
        });
    }
    for (i, &w) in sizes.iter().enumerate() {
        let d = dims[r - k + i];
        if w == 0 || w > d || strides[i] == 0 {
            return Err(ArrayError::ShapeMismatch {
                source: w,
                target: d,
            });
        }
    }
    Ok(())
}

/// `[((d_i - w_i)/s_i + 1)..., lead..., sizes...]` -- positions lead, the
/// non-windowed axes follow, the window sizes trail.
fn out_dims(dims: &[usize], sizes: &[usize], strides: &[usize]) -> Vec<usize> {
    let (r, k) = (dims.len(), sizes.len());
    let mut out: Vec<usize> = sizes
        .iter()
        .enumerate()
        .map(|(i, &w)| (dims[r - k + i] - w) / strides[i] + 1)
        .collect();
    out.extend_from_slice(&dims[..r - k]);
    out.extend_from_slice(sizes);
    out
}

/// Copy each output element from its source. Output axes are laid out as
/// `[positions (k), non-windowed (r-k), window (k)]`: non-windowed axis
/// `j` maps straight through from `oidx[k + j]`, and windowed axis `i`
/// reads input index `position*stride + offset` from `oidx[i]`/`oidx[r+i]`.
fn gather(
    data: &[f64],
    dims: &[usize],
    sizes: &[usize],
    strides: &[usize],
    out_dims: &[usize],
) -> Vec<f64> {
    let (r, k) = (dims.len(), sizes.len());
    let in_stride = row_major_strides(dims);
    let out_stride = row_major_strides(out_dims);
    let total: usize = out_dims.iter().product::<usize>().max(1);
    (0..total)
        .map(|o| {
            let mut rem = o;
            let oidx: Vec<usize> = out_stride
                .iter()
                .map(|&s| {
                    let idx = rem / s;
                    rem %= s;
                    idx
                })
                .collect();
            let mut in_flat = 0;
            for (j, s) in in_stride.iter().enumerate().take(r - k) {
                in_flat += oidx[k + j] * s;
            }
            for i in 0..k {
                let (pos, off) = (oidx[i], oidx[r + i]);
                in_flat += (pos * strides[i] + off) * in_stride[r - k + i];
            }
            data[in_flat]
        })
        .collect()
}

/// Row-major strides for a shape (trailing axis has stride 1).
fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for j in (0..dims.len().saturating_sub(1)).rev() {
        s[j] = s[j + 1] * dims[j + 1];
    }
    s
}
