//! Autograd substrate: the [`Tape`] recording structure, its node
//! types, the elementwise op kernels, and the pure gradient kernels
//! shared by the backward pass.

pub mod grad_kernels;
pub mod kernels_softmax;
pub mod ops;
pub mod resident;
pub mod resident_backward;
pub mod tape;

pub use kernels_softmax::{accumulate, seed_ones, softmax_backward, softmax_forward};
pub use ops::{BinaryOp, UnaryOp};
pub use resident::{ResidentReq, map_binary, map_unary, try_resident};
pub use tape::{NodeData, NodeId, NodeKind, Tape};
