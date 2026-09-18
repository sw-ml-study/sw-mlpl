//! Leaf backward wrappers for the structural ops (patchify / concat / stack /
//! take): each forces the upstream to a dense array, runs the exact CPU
//! kernel from `mlpl_autograd_tape::grad_kernels`, and accumulates into the
//! parent(s). Split from `backward_shape.rs` for the module function budget.

use mlpl_array::{DenseArray, Shape};

use mlpl_autograd_tape::grad_kernels::{
    concat_backward, patchify_backward, stack_backward, take_backward,
};
use mlpl_autograd_tape::{NodeId, Tape, accumulate};

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
