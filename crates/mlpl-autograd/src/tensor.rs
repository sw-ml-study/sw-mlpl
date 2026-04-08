//! Differentiable tensor handle.

use std::rc::Rc;

use mlpl_array::DenseArray;

use crate::backward;
use crate::tape::{NodeData, NodeId, NodeKind, Tape};

/// A differentiable tensor: a handle into a [`Tape`] node.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) node: NodeId,
    pub(crate) tape: Rc<Tape>,
}

impl Tensor {
    /// Construct a leaf tensor on `tape`.
    #[must_use]
    pub fn leaf(tape: Rc<Tape>, value: DenseArray, requires_grad: bool) -> Self {
        let node = tape.push(NodeData {
            value,
            grad: None,
            kind: NodeKind::Leaf,
            requires_grad,
        });
        Self { node, tape }
    }

    /// Construct a trainable parameter leaf (`requires_grad` = true).
    #[must_use]
    pub fn param(tape: Rc<Tape>, value: DenseArray) -> Self {
        Self::leaf(tape, value, true)
    }

    /// Node id on the tape.
    #[must_use]
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Clone of the forward value.
    #[must_use]
    pub fn value(&self) -> DenseArray {
        self.tape.nodes()[self.node.0].value.clone()
    }

    /// Clone of the accumulated gradient, if any.
    #[must_use]
    pub fn grad(&self) -> Option<DenseArray> {
        self.tape.nodes()[self.node.0].grad.clone()
    }

    /// Whether this node is a trainable leaf.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.tape.nodes()[self.node.0].requires_grad
    }

    /// Run reverse-mode backward from this tensor.
    pub fn backward(&self) {
        backward::backward(&self.tape, self.node);
    }
}
