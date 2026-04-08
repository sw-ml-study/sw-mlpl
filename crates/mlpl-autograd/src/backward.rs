//! Reverse-mode backward pass.

use mlpl_array::DenseArray;

use crate::ops::{accumulate, unbroadcast};
use crate::tape::{NodeId, NodeKind, Tape};

/// Run a backward pass rooted at `root`, seeding `d root / d root = 1`.
///
/// Walks nodes in reverse topological order (descending node id is
/// safe since ops always append after their parents).
pub fn backward(tape: &Tape, root: NodeId) {
    seed_root(tape, root);
    let len = tape.len();
    for idx in (0..len).rev() {
        propagate(tape, NodeId(idx));
    }
}

fn seed_root(tape: &Tape, root: NodeId) {
    let mut nodes = tape.nodes_mut();
    let node = &mut nodes[root.0];
    if node.grad.is_none() {
        let ones = vec![1.0; node.value.data().len()];
        node.grad = Some(DenseArray::new(node.value.shape().clone(), ones).expect("shape"));
    }
}

fn propagate(tape: &Tape, id: NodeId) {
    // Snapshot what we need without holding the borrow across mutation.
    let (kind, upstream, parents_values) = {
        let nodes = tape.nodes();
        let node = &nodes[id.0];
        let Some(upstream) = node.grad.clone() else {
            return;
        };
        let parents_values = match &node.kind {
            NodeKind::Leaf => None,
            NodeKind::Unary { parent, .. } => {
                Some((nodes[parent.0].value.clone(), DenseArray::from_scalar(0.0)))
            }
            NodeKind::Binary { left, right, .. } => {
                Some((nodes[left.0].value.clone(), nodes[right.0].value.clone()))
            }
        };
        (node.kind.clone(), upstream, parents_values)
    };

    match kind {
        NodeKind::Leaf => {}
        NodeKind::Unary { op, parent } => {
            let (x, _) = parents_values.expect("unary parent");
            let y = tape.nodes()[id.0].value.clone();
            let grad_x = op.backward(&x, &y, &upstream);
            let target_shape = tape.nodes()[parent.0].value.shape().clone();
            let grad_x = unbroadcast(grad_x, &target_shape);
            let mut nodes = tape.nodes_mut();
            accumulate(&mut nodes[parent.0].grad, grad_x);
        }
        NodeKind::Binary { op, left, right } => {
            let (a, b) = parents_values.expect("binary parents");
            let (ga, gb) = op.backward(&a, &b, &upstream);
            let left_shape = tape.nodes()[left.0].value.shape().clone();
            let right_shape = tape.nodes()[right.0].value.shape().clone();
            let ga = unbroadcast(ga, &left_shape);
            let gb = unbroadcast(gb, &right_shape);
            let mut nodes = tape.nodes_mut();
            accumulate(&mut nodes[left.0].grad, ga);
            accumulate(&mut nodes[right.0].grad, gb);
        }
    }
}
