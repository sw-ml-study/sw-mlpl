//! Reverse-mode backward pass.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_matmul::prelude::*;

use crate::ops::{accumulate, softmax_backward};
use crate::tape::{NodeId, NodeKind, Tape};

/// Run a backward pass rooted at `root`, seeding `d root / d root = 1`.
///
/// Walks nodes in reverse topological order (descending node id is
/// safe since ops always append after their parents).
pub fn backward(tape: &Tape, root: NodeId) {
    {
        let mut nodes = tape.nodes_mut();
        let node = &mut nodes[root.0];
        if node.grad.is_none() {
            let ones = vec![1.0; node.value.data().len()];
            node.grad = Some(DenseArray::new(node.value.shape().clone(), ones).expect("shape"));
        }
    }
    let len = tape.len();
    for idx in (0..len).rev() {
        propagate(tape, NodeId(idx));
    }
}

fn unbroadcast(grad: DenseArray, target_shape: &Shape) -> DenseArray {
    if grad.shape() == target_shape {
        return grad;
    }
    if target_shape.rank() == 0 {
        let s: f64 = grad.data().iter().sum();
        return DenseArray::from_scalar(s);
    }
    grad
}

fn propagate(tape: &Tape, id: NodeId) {
    let (kind, upstream) = {
        let nodes = tape.nodes();
        let node = &nodes[id.0];
        let Some(upstream) = node.grad.clone() else {
            return;
        };
        (node.kind.clone(), upstream)
    };

    match kind {
        NodeKind::Leaf => {}
        NodeKind::Unary { op, parent } => prop_unary(tape, id, parent, op, &upstream),
        NodeKind::Binary { op, left, right } => prop_binary(tape, left, right, op, &upstream),
        NodeKind::SumAll { parent } => prop_sum_mean(tape, parent, &upstream, false),
        NodeKind::MeanAll { parent } => prop_sum_mean(tape, parent, &upstream, true),
        NodeKind::Softmax { parent, axis } => {
            let y = tape.nodes()[id.0].value.clone();
            let grad = softmax_backward(&y, &upstream, axis);
            accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
        }
        NodeKind::Transpose { parent } => {
            let grad = upstream.transpose();
            accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
        }
        NodeKind::Reshape { parent, orig_shape } => {
            let grad = upstream.reshape(orig_shape).expect("reshape back");
            accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
        }
        NodeKind::MatMul { left, right } => prop_matmul(tape, left, right, &upstream),
        NodeKind::CrossEntropy { logits, targets } => {
            let logits_val = tape.nodes()[logits.0].value.clone();
            let g = upstream.data()[0];
            let grad = crate::tensor_ops::cross_entropy_backward(&logits_val, &targets, g);
            accumulate(&mut tape.nodes_mut()[logits.0].grad, grad);
        }
        NodeKind::Patchify {
            parent,
            orig_shape,
            patch_size,
        } => {
            let g = patchify_backward(&upstream, &orig_shape, patch_size);
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
        NodeKind::Concat {
            left,
            right,
            axis,
            left_size,
        } => {
            let (ga, gb) = concat_backward(&upstream, axis, left_size);
            let mut nodes = tape.nodes_mut();
            accumulate(&mut nodes[left.0].grad, ga);
            accumulate(&mut nodes[right.0].grad, gb);
        }
        NodeKind::Stack {
            parents,
            axis,
            parent_size_along_axis,
        } => {
            let grads = stack_backward(&upstream, parents.len(), axis, parent_size_along_axis);
            let mut nodes = tape.nodes_mut();
            for (pid, g) in parents.iter().zip(grads) {
                accumulate(&mut nodes[pid.0].grad, g);
            }
        }
        NodeKind::Take {
            parent,
            orig_shape,
            axis,
            idx,
        } => {
            let g = take_backward(&upstream, &orig_shape, axis, idx);
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
    }
}

fn take_backward(upstream: &DenseArray, orig_shape: &Shape, axis: usize, idx: usize) -> DenseArray {
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

fn patchify_backward(upstream: &DenseArray, orig_shape: &Shape, p: usize) -> DenseArray {
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

fn stack_backward(
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

fn concat_backward(
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
    let left = DenseArray::new(Shape::new(ld), la).expect("shape");
    let right = DenseArray::new(Shape::new(rd), rb).expect("shape");
    (left, right)
}

fn prop_unary(
    tape: &Tape,
    id: NodeId,
    parent: NodeId,
    op: crate::ops::UnaryOp,
    upstream: &DenseArray,
) {
    let x = tape.nodes()[parent.0].value.clone();
    let y = tape.nodes()[id.0].value.clone();
    let grad_x = op.backward(&x, &y, upstream);
    let target_shape = tape.nodes()[parent.0].value.shape().clone();
    let grad_x = unbroadcast(grad_x, &target_shape);
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad_x);
}

fn prop_binary(
    tape: &Tape,
    left: NodeId,
    right: NodeId,
    op: crate::ops::BinaryOp,
    upstream: &DenseArray,
) {
    let a = tape.nodes()[left.0].value.clone();
    let b = tape.nodes()[right.0].value.clone();
    let (ga, gb) = op.backward(&a, &b, upstream);
    let left_shape = tape.nodes()[left.0].value.shape().clone();
    let right_shape = tape.nodes()[right.0].value.shape().clone();
    let ga = unbroadcast(ga, &left_shape);
    let gb = unbroadcast(gb, &right_shape);
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}

fn prop_sum_mean(tape: &Tape, parent: NodeId, upstream: &DenseArray, mean: bool) {
    let parent_shape = tape.nodes()[parent.0].value.shape().clone();
    let n = parent_shape.elem_count();
    let g = if mean {
        upstream.data()[0] / n as f64
    } else {
        upstream.data()[0]
    };
    let grad = DenseArray::new(parent_shape, vec![g; n]).expect("shape");
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

fn prop_matmul(tape: &Tape, left: NodeId, right: NodeId, upstream: &DenseArray) {
    let a = tape.nodes()[left.0].value.clone();
    let b = tape.nodes()[right.0].value.clone();
    // grad_a = upstream @ b^T (or outer(upstream, b) for matrix-vector);
    // grad_b = a^T @ upstream.
    let ga = if b.shape().rank() == 1 {
        let m = upstream.shape().dims()[0];
        let k = b.shape().dims()[0];
        let mut data = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                data[i * k + j] = upstream.data()[i] * b.data()[j];
            }
        }
        DenseArray::new(Shape::new(vec![m, k]), data).expect("shape")
    } else {
        upstream
            .matmul(&b.transpose())
            .expect("matmul upstream b^T")
    };
    let gb = a.transpose().matmul(upstream).expect("matmul grad_b");
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}
