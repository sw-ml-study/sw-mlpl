//! Autograd substrate: the [`Tape`] recording structure, its node
//! types, the elementwise op kernels, and the pure gradient kernels
//! shared by the backward pass.

pub mod grad_kernels;
pub mod kernels_softmax;
pub mod ops;
pub mod tape;

pub use kernels_softmax::{accumulate, softmax_backward, softmax_forward};
pub use ops::{BinaryOp, UnaryOp};
pub use tape::{NodeData, NodeId, NodeKind, Tape};
