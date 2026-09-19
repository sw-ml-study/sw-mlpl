//! Tape-touching backward wrappers for the shape ops; the pure
//! kernels they call live in `crate::grad_kernels` / `crate::grad_kernels_shape`.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_shape::prelude::*;

use crate::grad_kernels::{reduce_sum_backward, windows_backward};
use mlpl_array_ops_compose::prelude::RotateExt;
use mlpl_autograd_tape::{NodeId, NodeKind, Tape, accumulate, accumulate_pair, resident};
use mlpl_tensor_handle::{SeamEvent, TensorHandle, bump_if};

use crate::backward_shape_kernels::{prop_concat, prop_patchify, prop_stack, prop_take};

/// Dispatch the structural node kinds (everything that is a pure
/// re-arrangement or the cross-entropy fused loss). The
/// elementwise / linalg / reduction kinds stay in
/// `backward::propagate`; it forwards every other kind here.
/// Transpose and reshape gradients stay lazy on a resident tape;
/// the remaining kinds force the upstream and run the exact CPU
/// kernels (counted as fallbacks when the tape is resident).
pub(crate) fn propagate_shape(tape: &Tape, kind: NodeKind, upstream: &TensorHandle) {
    match kind {
        NodeKind::Transpose { parent, perm } => {
            let g = match &perm {
                None => resident::transpose_backward(tape, upstream)
                    .unwrap_or_else(|| upstream.to_dense().transpose().into()),
                Some(p) => {
                    // Backward of a permutation is the permutation by its
                    // inverse: inv[p[i]] = i.
                    let mut inv = vec![0usize; p.len()];
                    for (i, &pi) in p.iter().enumerate() {
                        inv[pi] = i;
                    }
                    upstream
                        .to_dense()
                        .transpose_axes(&inv)
                        .expect("inverse of a valid permutation")
                        .into()
                }
            };
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
        NodeKind::Windows {
            parent,
            orig_shape,
            sizes,
            strides,
        } => {
            let g = windows_backward(upstream, &orig_shape, &sizes, &strides);
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
        NodeKind::ReduceSum {
            parent,
            orig_shape,
            axes,
        } => {
            let g = reduce_sum_backward(upstream, &orig_shape, &axes);
            accumulate(&mut tape.nodes_mut()[parent.0].grad, g);
        }
        _ => unreachable!("non-structural kinds are handled in backward::propagate"),
    }
}

fn prop_reshape(tape: &Tape, parent: NodeId, orig_shape: &Shape, upstream: &DenseArray) {
    let grad = upstream.reshape(orig_shape.clone()).expect("reshape back");
    accumulate(&mut tape.nodes_mut()[parent.0].grad, grad);
}

/// Backward of `x^exp` (elementwise): `grad = upstream * exp * x^(exp-1)`,
/// read against the parent's forward value `x`.
pub(crate) fn prop_pow_const(tape: &Tape, parent: NodeId, exp: f64, upstream: &TensorHandle) {
    let x = tape.nodes()[parent.0].value.to_dense();
    let up = upstream.to_dense();
    let grad: Vec<f64> = x
        .data()
        .iter()
        .zip(up.data())
        .map(|(&xi, &g)| g * exp * xi.powf(exp - 1.0))
        .collect();
    let grad = DenseArray::new(x.shape().clone(), grad).expect("shape preserved");
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
    let grad = crate::cross_entropy::cross_entropy_backward(&logits_val, targets, g);
    accumulate(&mut tape.nodes_mut()[logits.0].grad, grad);
}
