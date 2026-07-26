//! Reverse-mode backward pass.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_matmul::prelude::*;
use mlpl_array_ops_shape::prelude::*;

use mlpl_autograd_tape::grad_kernels::unbroadcast;
use mlpl_autograd_tape::{NodeId, NodeKind, Tape, accumulate, softmax_backward};

use crate::backward_shape::propagate_shape;

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
        NodeKind::Softmax { parent, axis } => prop_softmax(tape, id, parent, axis, &upstream),
        NodeKind::MatMul { left, right } => prop_matmul(tape, left, right, &upstream),
        // Structural kinds (transpose/reshape/cross-entropy/patchify/
        // concat/stack/take/rotate) dispatch in backward_shape.rs.
        other => propagate_shape(tape, other, &upstream),
    }
}

fn prop_softmax(tape: &Tape, id: NodeId, parent: NodeId, axis: usize, upstream: &DenseArray) {
    let y = tape.nodes()[id.0].value.clone();
    let grad = softmax_backward(&y, upstream, axis);
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

fn prop_unary(
    tape: &Tape,
    id: NodeId,
    parent: NodeId,
    op: mlpl_autograd_tape::UnaryOp,
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
    op: mlpl_autograd_tape::BinaryOp,
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
