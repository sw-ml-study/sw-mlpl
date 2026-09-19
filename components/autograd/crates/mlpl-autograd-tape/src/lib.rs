//! Autograd substrate: the [`Tape`] recording structure, its node
//! types, and the elementwise op kernels. The pure gradient kernels for
//! the backward pass live in the `mlpl-autograd-backward` crate.

pub mod kernels_softmax;
pub mod ops;
pub mod resident;
pub mod resident_backward;
pub mod tape;

pub use kernels_softmax::{
    accumulate, accumulate_pair, seed_ones, softmax_backward, softmax_forward,
};
pub use ops::{BinaryOp, UnaryOp, map_binary, map_unary};
pub use resident::{ResidentReq, try_resident};
pub use tape::{NodeData, NodeId, NodeKind, Tape};
