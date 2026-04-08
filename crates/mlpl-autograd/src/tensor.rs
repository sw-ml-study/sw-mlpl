//! Differentiable tensor wrapper.

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use mlpl_array::DenseArray;

use crate::tape::{Node, NodeId, Tape};

/// A differentiable tensor.
///
/// Holds a value, an optional accumulated gradient, a handle to the
/// tape node that produced it, and a `requires_grad` flag.
#[derive(Debug)]
pub struct Tensor {
    value: DenseArray,
    grad: RefCell<Option<DenseArray>>,
    node: NodeId,
    requires_grad: bool,
    tape: Rc<Tape>,
}

impl Tensor {
    /// Construct a leaf tensor on the given tape.
    #[must_use]
    pub fn leaf(tape: Rc<Tape>, value: DenseArray, requires_grad: bool) -> Self {
        let node = tape.push(Node::Leaf { requires_grad });
        Self {
            value,
            grad: RefCell::new(None),
            node,
            requires_grad,
            tape,
        }
    }

    /// Construct a trainable parameter leaf (`requires_grad` = true).
    #[must_use]
    pub fn param(tape: Rc<Tape>, value: DenseArray) -> Self {
        Self::leaf(tape, value, true)
    }

    /// Borrow the underlying value.
    #[must_use]
    pub fn value(&self) -> &DenseArray {
        &self.value
    }

    /// Borrow the accumulated gradient, if any.
    #[must_use]
    pub fn grad(&self) -> Ref<'_, Option<DenseArray>> {
        self.grad.borrow()
    }

    /// Whether this tensor participates in gradient tracking.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// The tape node id that produced this tensor.
    #[must_use]
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Reverse-mode backward pass from this tensor.
    ///
    /// For the scaffold this is a no-op on leaves. Non-leaf behavior is
    /// introduced in later steps as operations are added to the tape.
    pub fn backward(&self) {
        let nodes = self.tape.nodes.borrow();
        if let Some(Node::Leaf { .. }) = nodes.get(self.node.0) {
            // Leaves have no parents; nothing to propagate.
        }
    }
}
