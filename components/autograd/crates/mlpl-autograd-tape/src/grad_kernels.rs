//! Pure gradient kernels for the shape ops: DenseArray in,
//! DenseArray out, no tape access. The thin tape-touching `prop_*`
//! wrappers live with the backward pass in mlpl-autograd.

use mlpl_array::{DenseArray, Shape};

pub fn unbroadcast(grad: DenseArray, target_shape: &Shape) -> DenseArray {
    if grad.shape() == target_shape {
        return grad;
    }
    if target_shape.rank() == 0 {
        let s: f64 = grad.data().iter().sum();
        return DenseArray::from_scalar(s);
    }
    grad
}

pub fn take_backward(
    upstream: &DenseArray,
    orig_shape: &Shape,
    axis: usize,
    idx: usize,
) -> DenseArray {
    let dims = orig_shape.dims();
    let outer: usize = dims[..axis].iter().product();
    let axis_size = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let mut out = vec![0.0; orig_shape.elem_count()];
    let up = upstream.data();
    for o in 0..outer {
        let src = o * inner;
        let dst = (o * axis_size + idx) * inner;
        out[dst..dst + inner].copy_from_slice(&up[src..src + inner]);
    }
    DenseArray::new(orig_shape.clone(), out).expect("shape")
}

pub fn patchify_backward(upstream: &DenseArray, orig_shape: &Shape, p: usize) -> DenseArray {
    let dims = orig_shape.dims();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let (nh, nw, patch_len) = (h / p, w / p, p * p * c);
    let up = upstream.data();
    let mut out = vec![0.0; b * c * h * w];
    for b_i in 0..b {
        for i in 0..nh {
            for j in 0..nw {
                let n = i * nw + j;
                for c_i in 0..c {
                    for dy in 0..p {
                        for dx in 0..p {
                            let dst = ((b_i * c + c_i) * h + i * p + dy) * w + j * p + dx;
                            let src = (b_i * nh * nw + n) * patch_len + c_i * p * p + dy * p + dx;
                            out[dst] = up[src];
                        }
                    }
                }
            }
        }
    }
    DenseArray::new(orig_shape.clone(), out).expect("shape")
}

pub fn stack_backward(
    upstream: &DenseArray,
    n: usize,
    axis: usize,
    parent_size: usize,
) -> Vec<DenseArray> {
    let dims = upstream.shape().dims();
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let parent_stride = parent_size * inner;
    let mut parent_dims = dims.to_vec();
    parent_dims[axis] = parent_size;
    let parent_elems: usize = parent_dims.iter().product();
    let up = upstream.data();
    let mut outs: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(parent_elems)).collect();
    for o in 0..outer {
        for (k, out) in outs.iter_mut().enumerate() {
            let src = (o * n + k) * parent_stride;
            out.extend_from_slice(&up[src..src + parent_stride]);
        }
    }
    outs.into_iter()
        .map(|v| DenseArray::new(Shape::new(parent_dims.clone()), v).expect("shape"))
        .collect()
}

pub fn concat_backward(
    upstream: &DenseArray,
    axis: usize,
    left_size: usize,
) -> (DenseArray, DenseArray) {
    let dims = upstream.shape().dims();
    let right_size = dims[axis] - left_size;
    let mut ld = dims.to_vec();
    ld[axis] = left_size;
    let mut rd = dims.to_vec();
    rd[axis] = right_size;
    // Saga 30 step 002: walk the outer dims (dims[..axis]) and,
    // for each outer position, peel off `left_size * inner` then
    // `right_size * inner` elements from the upstream gradient.
    // `inner` is the product of dims after `axis` (1 for last axis).
    // Subsumes the original axis-0 / axis-1 branches: at axis=0 the
    // outer is 1 (empty-product) and the loop runs once on the
    // whole upstream slab; at axis=1 the outer is dims[0] and
    // inner is product(dims[2..]).
    let (la, rb) = split_chunks(upstream, dims, axis, left_size, right_size);
    let left = DenseArray::new(Shape::new(ld), la).expect("shape");
    let right = DenseArray::new(Shape::new(rd), rb).expect("shape");
    (left, right)
}

/// Walk the outer dims, peeling `left_size * inner` then
/// `right_size * inner` elements per outer position.
fn split_chunks(
    upstream: &DenseArray,
    dims: &[usize],
    axis: usize,
    left_size: usize,
    right_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let a_chunk = left_size * inner;
    let b_chunk = right_size * inner;
    let up = upstream.data();
    let mut la = Vec::with_capacity(outer * a_chunk);
    let mut rb = Vec::with_capacity(outer * b_chunk);
    for o in 0..outer {
        let row = o * (a_chunk + b_chunk);
        la.extend_from_slice(&up[row..row + a_chunk]);
        rb.extend_from_slice(&up[row + a_chunk..row + a_chunk + b_chunk]);
    }
    (la, rb)
}
