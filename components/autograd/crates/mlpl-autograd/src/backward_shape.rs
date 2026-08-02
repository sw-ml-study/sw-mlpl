//! Tape-touching backward wrappers for the shape ops; the pure
//! kernels they call live in `mlpl_autograd_tape::grad_kernels`.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_shape::prelude::*;

use mlpl_array_ops_compose::prelude::RotateExt;
use mlpl_autograd_tape::grad_kernels::{
    concat_backward, patchify_backward, stack_backward, take_backward,
};
use mlpl_autograd_tape::{NodeId, NodeKind, Tape, accumulate, accumulate_pair, resident};
use mlpl_tensor_handle::{SeamEvent, TensorHandle, bump_if};

/// Dispatch the structural node kinds (everything that is a pure
/// re-arrangement or the cross-entropy fused loss). The
/// elementwise / linalg / reduction kinds stay in
/// `backward::propagate`; it forwards every other kind here.
/// Transpose and reshape gradients stay lazy on a resident tape;
/// the remaining kinds force the upstream and run the exact CPU
/// kernels (counted as fallbacks when the tape is resident).
pub(crate) fn propagate_shape(tape: &Tape, kind: NodeKind, upstream: &TensorHandle) {
    match kind {
        NodeKind::Transpose { parent } => {
            let g = resident::transpose_backward(tape, upstream)
                .unwrap_or_else(|| upstream.to_dense().transpose().into());
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
        NodeKind::Reshape { parent, orig_shape } => {
            match resident::reshape_backward(tape, upstream, orig_shape.dims()) {
                Some(g) => accumulate(&mut tape.nodes_mut()[parent.0].grad, g),
                None => prop_reshape(tape, parent, &orig_shape, &upstream.to_dense()),
            }
        }
        NodeKind::Concat {
            left,
            right,
            axis,
            left_size,
        } => {
            let dev = tape
                .resident
                .get()
                .then(|| resident::as_dev(upstream))
                .flatten()
                .and_then(|up| up.dev_split2(axis, left_size).ok());
            match dev {
                Some((ga, gb)) => accumulate_pair(tape, left, right, ga, gb),
                None => prop_concat(tape, left, right, axis, left_size, &upstream.to_dense()),
            }
        }
        other => propagate_dense(tape, other, &upstream.to_dense()),
    }
}

/// The kinds whose backward runs on the exact CPU kernels.
fn propagate_dense(tape: &Tape, kind: NodeKind, upstream: &DenseArray) {
    bump_if(tape.resident.get(), SeamEvent::CpuFallback);
    match kind {
        NodeKind::CrossEntropy { logits, targets } => {
            prop_cross_entropy(tape, logits, &targets, upstream);
        }
        NodeKind::Patchify {
            parent,
            orig_shape,
            patch_size,
        } => prop_patchify(tape, parent, &orig_shape, patch_size, upstream),
        NodeKind::Stack {
            parents,
            axis,
            parent_size_along_axis,
        } => prop_stack(tape, &parents, axis, parent_size_along_axis, upstream),
        NodeKind::Take {
            parent,
            orig_shape,
            axis,
            idx,
        } => prop_take(tape, parent, &orig_shape, axis, idx, upstream),
        NodeKind::Rotate { parent, k, axis } => {
            let g = upstream
                .rotate(-k, axis)
                .expect("rotate grad: axis in range");
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
        _ => unreachable!("non-structural kinds are handled in backward::propagate"),
    }
}

fn prop_reshape(tape: &Tape, parent: NodeId, orig_shape: &Shape, upstream: &DenseArray) {
    let grad = upstream.reshape(orig_shape.clone()).expect("reshape back");
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

pub(crate) fn prop_cross_entropy(
    tape: &Tape,
    logits: NodeId,
    targets: &[usize],
    upstream: &DenseArray,
) {
    let logits_val = tape.nodes()[logits.0].value.to_dense();
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
