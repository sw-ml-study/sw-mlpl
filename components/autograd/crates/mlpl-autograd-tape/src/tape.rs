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
    /// Fused log-softmax + NLL over integer targets.
    ///
    /// Forward: scalar `-mean(log_softmax(logits)[i, targets[i]])`.
    /// Targets are owned by the node because they are integer-valued
    /// and do not flow gradient; storing them here keeps the tape
    /// self-contained for the backward pass.
    CrossEntropy {
        /// Logits parent (rank-2 `[N, V]` or rank-3 `[B, T, V]`).
        logits: NodeId,
        /// Flattened target indices, one per row in `[N, V]`
        /// (or one per `(b, t)` pair in `[B, T, V]`).
        targets: Vec<usize>,
    },
    /// Saga 29 step 005: patchify `[B, C, H, W]` into
    /// `[B, N, P*P*C]` where `N = (H/P) * (W/P)`. The
    /// `orig_shape` of the parent is needed to scatter the
    /// gradient back to image space on backward.
    Patchify {
        /// Parent node id (the `[B, C, H, W]` image batch).
        parent: NodeId,
        /// Original `[B, C, H, W]` shape -- needed because
        /// the parent's forward value is consulted only via
        /// node id, not shape, on the backward path.
        orig_shape: Shape,
        /// Square patch side length `P`.
        patch_size: usize,
    },
    /// Saga 29 step 005: concat two parents along `axis`.
    /// `left_size` is the size of `left` along `axis` so the
    /// backward can split the gradient correctly without
    /// re-reading the parent's value.
    Concat {
        /// Left parent id.
        left: NodeId,
        /// Right parent id.
        right: NodeId,
        /// Axis of concatenation (0 or 1 in this initial
        /// release).
        axis: usize,
        /// Size of `left` along `axis`; the split point in
        /// the gradient.
        left_size: usize,
    },
    /// Saga 29 step 013: N-way stack of like-shaped parents
    /// along an existing axis. Forward concatenates each
    /// parent's slab; backward splits the upstream gradient
    /// into N equal-size slabs and delivers each to its
    /// parent. Replaces the prior O(N^2) binary-concat chain
    /// in the multi-head and rank-3-batched attention paths.
    Stack {
        /// Parent node ids in stack order.
        parents: Vec<NodeId>,
        /// Axis along which to stack.
        axis: usize,
        /// Size of each parent along `axis`. All parents share
        /// this value (validated at construction).
        parent_size_along_axis: usize,
    },
    /// Saga 29 step 007: drop one axis at a single integer
    /// index. The `orig_shape` is the parent's shape;
    /// backward scatters the upstream gradient into a zero-
    /// filled array of `orig_shape`, placing the gradient at
    /// `axis = idx`.
    Take {
        /// Parent node id.
        parent: NodeId,
        /// Parent's original shape so the backward can
        /// scatter into the right buffer.
        orig_shape: Shape,
        /// Axis to drop.
        axis: usize,
        /// Index along `axis` to extract.
        idx: usize,
    },
    /// Cyclic rotate along an axis (Game of Life saga step 1).
    /// A pure permutation: backward rotates the upstream
    /// gradient by `-k` along the same axis.
    Rotate {
        /// Parent node id.
        parent: NodeId,
        /// Rotation amount (positive = element `k` to the front).
        k: i64,
        /// Axis rotated.
        axis: usize,
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
