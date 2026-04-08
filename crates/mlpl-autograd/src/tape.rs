//! Tape and node types for the autograd engine.

use std::cell::RefCell;
use std::rc::Rc;

/// Opaque identifier for a node on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A node on the tape. For the scaffold step only leaves exist.
#[derive(Debug)]
pub enum Node {
    /// A leaf tensor with no parents.
    Leaf {
        /// Whether this leaf participates in gradient tracking.
        requires_grad: bool,
    },
}

/// A tape recording the computation graph for reverse-mode autograd.
#[derive(Debug, Default)]
pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
}

impl Tape {
    /// Create a fresh empty tape.
    #[must_use]
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            nodes: RefCell::new(Vec::new()),
        })
    }

    /// Push a node and return its unique id.
    pub fn push(&self, node: Node) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(node);
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
}
