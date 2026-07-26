//! Tape-touching backward wrappers for the shape ops; the pure
//! kernels they call live in `mlpl_autograd_tape::grad_kernels`.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_shape::prelude::*;

use mlpl_autograd_tape::grad_kernels::{
    concat_backward, patchify_backward, stack_backward, take_backward,
};
use mlpl_autograd_tape::{NodeId, Tape, accumulate};

pub(crate) fn prop_transpose(tape: &Tape, parent: NodeId, upstream: &DenseArray) {
    accumulate(&mut tape.nodes_mut()[parent.0].grad, upstream.transpose());
}

pub(crate) fn prop_reshape(tape: &Tape, parent: NodeId, orig_shape: &Shape, upstream: &DenseArray) {
    let grad = upstream.reshape(orig_shape.clone()).expect("reshape back");
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

pub(crate) fn prop_cross_entropy(
    tape: &Tape,
    logits: NodeId,
    targets: &[usize],
    upstream: &DenseArray,
) {
    let logits_val = tape.nodes()[logits.0].value.clone();
    let g = upstream.data()[0];
    let grad = crate::tensor_ops::cross_entropy_backward(&logits_val, targets, g);
    accumulate(&mut tape.nodes_mut()[logits.0].grad, grad);
}

pub(crate) fn prop_patchify(
    tape: &Tape,
    parent: NodeId,
    orig_shape: &Shape,
    patch_size: usize,
    upstream: &DenseArray,
) {
    let g = patchify_backward(upstream, orig_shape, patch_size);
    accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
}

pub(crate) fn prop_concat(
    tape: &Tape,
    left: NodeId,
    right: NodeId,
    axis: usize,
    left_size: usize,
    upstream: &DenseArray,
) {
    let (ga, gb) = concat_backward(upstream, axis, left_size);
    let mut nodes = tape.nodes_mut();
    accumulate(&mut nodes[left.0].grad, ga);
    accumulate(&mut nodes[right.0].grad, gb);
}

pub(crate) fn prop_stack(
    tape: &Tape,
    parents: &[NodeId],
    axis: usize,
    parent_size: usize,
    upstream: &DenseArray,
) {
    let grads = stack_backward(upstream, parents.len(), axis, parent_size);
    let mut nodes = tape.nodes_mut();
    for (pid, g) in parents.iter().zip(grads) {
        accumulate(&mut nodes[pid.0].grad, g);
    }
}

pub(crate) fn prop_take(
    tape: &Tape,
    parent: NodeId,
    orig_shape: &Shape,
    axis: usize,
    idx: usize,
    upstream: &DenseArray,
) {
    let g = take_backward(upstream, orig_shape, axis, idx);
    accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
}
