//! Reverse-mode autograd engine for MLPL.
//!
//! This crate provides a [`Tensor`] wrapper around
//! [`mlpl_array::DenseArray`] and a [`Tape`] that records operations
//! for reverse-mode differentiation.
//!
//! This is the v0.5 scaffold: leaf construction, unique node ids, and a
//! no-op `backward` on a leaf. Operations and gradient propagation land
//! in subsequent steps.

mod tape;
mod tensor;

pub use tape::{Node, NodeId, Tape};
pub use tensor::Tensor;
