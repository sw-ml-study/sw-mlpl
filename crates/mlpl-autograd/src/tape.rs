//! Tape storage for reverse-mode autograd.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};

use crate::ops::{BinaryOp, UnaryOp};

/// Opaque identifier for a node on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// What kind of node this is: a leaf, a unary op, or a binary op.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// A leaf tensor with no parents.
    Leaf,
    /// Result of a unary op applied to `parent`.
    Unary {
        /// Operation.
        op: UnaryOp,
        /// Parent node id.
        parent: NodeId,
    },
    /// Result of a binary op applied to `left` and `right`.
    Binary {
        /// Operation.
        op: BinaryOp,
        /// Left parent id.
        left: NodeId,
        /// Right parent id.
        right: NodeId,
    },
    /// Sum all elements of `parent` into a scalar.
    SumAll {
        /// Parent node id.
        parent: NodeId,
    },
    /// Mean over all elements of `parent` into a scalar.
    MeanAll {
        /// Parent node id.
        parent: NodeId,
    },
    /// Row-wise softmax along `axis` (last axis for rank-1 / rank-2).
    Softmax {
        /// Parent node id.
        parent: NodeId,
        /// Axis along which softmax is computed.
        axis: usize,
    },
    /// Transpose (reverse axes).
    Transpose {
        /// Parent node id.
        parent: NodeId,
    },
    /// Reshape to a new shape; `orig_shape` is the parent's original shape.
    Reshape {
        /// Parent node id.
        parent: NodeId,
        /// Parent's original shape, needed to un-reshape the gradient.
        orig_shape: Shape,
    },
    /// Matrix multiplication `left @ right`.
    MatMul {
        /// Left parent id.
        left: NodeId,
        /// Right parent id.
        right: NodeId,
    },
}

/// Per-node storage: the forward value, an accumulated gradient, the
/// node kind, and whether the node contributes gradients to leaves.
#[derive(Debug, Clone)]
pub struct NodeData {
    /// Forward value of this node.
    pub value: DenseArray,
    /// Accumulated gradient from backward passes, if any.
    pub grad: Option<DenseArray>,
    /// Kind: leaf, unary, or binary.
    pub kind: NodeKind,
    /// Whether this leaf is a trainable parameter.
    pub requires_grad: bool,
}

/// A tape recording the computation graph for reverse-mode autograd.
#[derive(Debug, Default)]
pub struct Tape {
    nodes: RefCell<Vec<NodeData>>,
}

impl Tape {
    /// Create a fresh empty tape.
    #[must_use]
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            nodes: RefCell::new(Vec::new()),
        })
    }

    /// Push a new node and return its id.
    pub fn push(&self, data: NodeData) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(data);
        id
    }

    /// Number of nodes currently on the tape.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Whether the tape is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    /// Borrow the node vector immutably.
    pub fn nodes(&self) -> Ref<'_, Vec<NodeData>> {
        self.nodes.borrow()
    }

    /// Borrow the node vector mutably.
    pub fn nodes_mut(&self) -> RefMut<'_, Vec<NodeData>> {
        self.nodes.borrow_mut()
    }
}
