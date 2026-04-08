//! Reverse-mode autograd engine for MLPL.
//!
//! Provides [`Tensor`], a handle into a [`Tape`] that records
//! elementwise operations and propagates gradients backward.

pub mod backward;
pub mod ops;
mod tape;
mod tensor;
mod tensor_ops;

pub use ops::{BinaryOp, UnaryOp};
pub use tape::{NodeData, NodeId, NodeKind, Tape};
pub use tensor::Tensor;
