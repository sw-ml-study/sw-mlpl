//! Pure gradient kernels (DenseArray in, DenseArray out, no tape access):
//! the matmul/elementwise/reduce family. The structural rearrangement
//! kernels live in `grad_kernels_shape`. Moved out of mlpl-autograd-tape by
//! the autograd-partition so the backward crate owns its kernels.

use mlpl_array::{DenseArray, Shape};

/// Outer product for the matrix-vector matmul backward:
/// `out[i, j] = upstream[i] * b[j]`.
#[must_use]
pub fn matvec_outer(upstream: &DenseArray, b: &DenseArray) -> DenseArray {
    let (m, k) = (upstream.shape().dims()[0], b.shape().dims()[0]);
    let mut data = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            data[i * k + j] = upstream.data()[i] * b.data()[j];
        }
    }
    DenseArray::new(Shape::new(vec![m, k]), data).expect("shape")
}

#[must_use]
pub fn unbroadcast(grad: DenseArray, target_shape: &Shape) -> DenseArray {
    if grad.shape() == target_shape {
        return grad;
    }
    // Sum over every broadcast axis: leading axes absent from the target
    // (prepended 1s) and axes the target holds at extent 1. This is the
    // backward of NumPy/APL broadcasting; without it a broadcast operand
    // keeps the larger output shape (subsumes the rank-0 sum-all case).
    let (g_dims, t_dims) = (grad.shape().dims(), target_shape.dims());
    let off = g_dims.len() - t_dims.len();
    let stride_of = |d: &[usize]| {
        let mut s = vec![1usize; d.len()];
        for j in (0..d.len().saturating_sub(1)).rev() {
            s[j] = s[j + 1] * d[j + 1];
        }
        s
    };
    let (g_stride, t_stride) = (stride_of(g_dims), stride_of(t_dims));
    let mut out = vec![0.0; target_shape.elem_count()];
    for (gi, &g) in grad.data().iter().enumerate() {
        let mut rem = gi;
        let mut tflat = 0;
        for (a, &gs) in g_stride.iter().enumerate() {
            let gidx = rem / gs;
            rem %= gs;
            if a >= off && t_dims[a - off] != 1 {
                tflat += gidx * t_stride[a - off];
            }
        }
        out[tflat] += g;
    }
    DenseArray::new(target_shape.clone(), out).expect("shape")
}

/// Scatter-add backward of `windows`: mirror the forward gather (map each
/// output flat index to its source input index) but ACCUMULATE the
/// upstream gradient there, since overlapping windows read a position
/// more than once. Writes into a zero-filled `orig_shape` buffer.
#[must_use]
pub fn windows_backward(
    upstream: &DenseArray,
    orig_shape: &Shape,
    sizes: &[usize],
    strides: &[usize],
) -> DenseArray {
    let dims = orig_shape.dims();
    let out_dims = upstream.shape().dims();
    let (r, k) = (dims.len(), sizes.len());
    let stride_of = |d: &[usize]| {
        let mut s = vec![1usize; d.len()];
        for j in (0..d.len().saturating_sub(1)).rev() {
            s[j] = s[j + 1] * d[j + 1];
        }
        s
    };
    let (in_stride, out_stride) = (stride_of(dims), stride_of(out_dims));
    let mut out = vec![0.0; orig_shape.elem_count()];
    for (o, &g) in upstream.data().iter().enumerate() {
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
            in_flat += (oidx[i] * strides[i] + oidx[r + i]) * in_stride[r - k + i];
        }
        out[in_flat] += g;
    }
    DenseArray::new(orig_shape.clone(), out).expect("shape")
}

/// Backward of a sum over `axes`: BROADCAST the upstream gradient (shaped
/// like the reduced array) back to `orig_shape`. Every input element that
/// summed into one output receives that output's gradient.
#[must_use]
pub fn reduce_sum_backward(
    upstream: &DenseArray,
    orig_shape: &Shape,
    axes: &[usize],
) -> DenseArray {
    let dims = orig_shape.dims();
    let mut reduced = axes.to_vec();
    reduced.sort_unstable();
    let reduced_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !reduced.contains(i))
        .map(|(_, &d)| d)
        .collect();
    let stride_of = |d: &[usize]| {
        let mut s = vec![1usize; d.len()];
        for j in (0..d.len().saturating_sub(1)).rev() {
            s[j] = s[j + 1] * d[j + 1];
        }
        s
    };
    let (orig_stride, red_stride) = (stride_of(dims), stride_of(&reduced_dims));
    let up = upstream.data();
    let mut out = vec![0.0; orig_shape.elem_count()];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut rem = o;
        let mut ridx = 0;
        let mut rk = 0;
        for (i, &s) in orig_stride.iter().enumerate() {
            let idx = rem / s;
            rem %= s;
            if !reduced.contains(&i) {
                ridx += idx * red_stride[rk];
                rk += 1;
            }
        }
        *slot = up[ridx];
    }
    DenseArray::new(orig_shape.clone(), out).expect("shape")
}
