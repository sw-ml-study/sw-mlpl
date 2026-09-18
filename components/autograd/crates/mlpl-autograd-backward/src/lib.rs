//! Reverse-mode backward pass for the MLPL autograd tape, split out of
//! `mlpl-autograd` (autograd-partition). Depends only on the tape substrate
//! and the array-ops kernels -- never on the `Tensor` API crate -- so the
//! dependency graph stays acyclic (array < tape < backward < autograd).

pub mod backward;
pub mod cross_entropy;

pub(crate) mod backward_shape;
mod backward_shape_kernels;

pub use backward::backward;
pub use cross_entropy::{cross_entropy_backward, cross_entropy_forward};
