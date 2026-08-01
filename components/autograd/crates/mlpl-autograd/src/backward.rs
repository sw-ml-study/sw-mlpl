//! Reverse-mode backward pass.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_matmul::prelude::*;
use mlpl_array_ops_shape::prelude::*;

use mlpl_autograd_tape::grad_kernels::unbroadcast;
use mlpl_autograd_tape::{NodeId, NodeKind, Tape, accumulate, resident_backward, softmax_backward};
use mlpl_tensor_handle::TensorHandle;

use crate::backward_shape::propagate_shape;

/// Run a backward pass rooted at `root`, seeding `d root / d root = 1`.
///
/// Walks nodes in reverse topological order (descending node id is
/// safe since ops always append after their parents).
pub fn backward(tape: &Tape, root: NodeId) {
    mlpl_autograd_tape::seed_ones(&mut tape.nodes_mut()[root.0]);
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
        // concat/stack/take/rotate) dispatch in backward_shape.rs on
        // the exact CPU kernels; the mixed-residency accumulate
        // re-uploads their grads when the rest of the tape is
        // resident.
        other => propagate_shape(tape, other, &upstream.to_dense()),
    }
}

fn prop_softmax(tape: &Tape, id: NodeId, parent: NodeId, axis: usize, upstream: &TensorHandle) {
    let y = tape.nodes()[id.0].value.clone();
    if tape.resident.get()
        && let Some(g) = resident_backward::softmax_backward(&y, upstream, axis)
    {
        return accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
    }
    let grad = softmax_backward(&y.to_dense(), &upstream.to_dense(), axis);
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

fn prop_unary(
    tape: &Tape,
    id: NodeId,
    parent: NodeId,
    op: mlpl_autograd_tape::UnaryOp,
    upstream: &TensorHandle,
) {
    let (xh, yh) = (
        tape.nodes()[parent.0].value.clone(),
        tape.nodes()[id.0].value.clone(),
    );
    if tape.resident.get()
        && xh.dims() == upstream.dims()
        && let Some(g) = resident_backward::unary_backward(op, &xh, &yh, upstream)
    {
        return accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
    }
    let (x, up) = (xh.to_dense(), upstream.to_dense());
    let grad_x = op.backward(&x, &yh.to_dense(), &up);
    let grad_x = unbroadcast(grad_x, &x.shape().clone());
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad_x);
}

fn prop_binary(
    tape: &Tape,
    left: NodeId,
    right: NodeId,
    op: mlpl_autograd_tape::BinaryOp,
    upstream: &TensorHandle,
) {
    let (ah, bh) = (
        tape.nodes()[left.0].value.clone(),
        tape.nodes()[right.0].value.clone(),
    );
    if tape.resident.get()
        && let Some((ga, gb)) = resident_backward::binary_backward(op, &ah, &bh, upstream)
    {
        let mut nodes = tape.nodes_mut();
        accumulate(&mut nodes[left.0].grad, ga);
        return accumulate(&mut nodes[right.0].grad, gb);
    }
    let (a, b, up) = (ah.to_dense(), bh.to_dense(), upstream.to_dense());
    let (ga, gb) = op.backward(&a, &b, &up);
    let (ga, gb) = (unbroadcast(ga, a.shape()), unbroadcast(gb, b.shape()));
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}

fn prop_sum_mean(tape: &Tape, parent: NodeId, upstream: &TensorHandle, mean: bool) {
    let dims = tape.nodes()[parent.0].value.dims();
    if tape.resident.get()
        && let Some(g) = resident_backward::sum_mean_backward(&dims, upstream, mean)
    {
        return accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
    }
    let parent_shape = Shape::new(dims);
    let n = parent_shape.elem_count();
    let up = upstream.to_dense();
    let g = if mean {
        up.data()[0] / n as f64
    } else {
        up.data()[0]
    };
    let grad = DenseArray::new(parent_shape, vec![g; n]).expect("shape");
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

fn prop_matmul(tape: &Tape, left: NodeId, right: NodeId, upstream: &TensorHandle) {
    let (ah, bh) = (
        tape.nodes()[left.0].value.clone(),
        tape.nodes()[right.0].value.clone(),
    );
    if tape.resident.get()
        && let Some((ga, gb)) = resident_backward::matmul_backward(&ah, &bh, upstream)
    {
        let mut nodes = tape.nodes_mut();
        accumulate(&mut nodes[left.0].grad, ga);
        return accumulate(&mut nodes[right.0].grad, gb);
    }
    let (a, b, up) = (ah.to_dense(), bh.to_dense(), upstream.to_dense());
    // grad_a = upstream @ b^T (or outer(upstream, b) for matrix-vector);
    // grad_b = a^T @ upstream.
    let ga = if b.shape().rank() == 1 {
        mlpl_autograd_tape::grad_kernels::matvec_outer(&up, &b)
    } else {
        up.matmul(&b.transpose()).expect("matmul upstream b^T")
    };
    let gb = a.transpose().matmul(&up).expect("matmul grad_b");
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}
