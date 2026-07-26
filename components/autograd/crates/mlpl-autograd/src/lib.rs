//! Reverse-mode autograd engine for MLPL.
//!
//! Provides [`Tensor`], a handle into a [`Tape`] that records
//! elementwise operations and propagates gradients backward. The
//! substrate (tape, op kernels, pure gradient kernels) lives in
//! `mlpl-autograd-tape` and is re-exported here so callers keep the
//! `mlpl_autograd::` paths.

pub mod backward;
mod backward_shape;
mod tensor;
mod tensor_ops;
mod tensor_reduce;
mod tensor_shape;

pub use mlpl_autograd_tape::ops;
pub use mlpl_autograd_tape::{NodeData, NodeId, NodeKind, Tape, softmax_backward, softmax_forward};
pub use ops::{BinaryOp, UnaryOp};
pub use tensor::Tensor;
