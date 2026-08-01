//! The backend contract: an opaque resident array plus the op
//! surface the autograd tape needs on BOTH the forward and the
//! backward pass. Kind enums keep the trait narrow (one method per
//! op family, not per op).

use std::any::Any;
use std::sync::Arc;

use mlpl_array::DenseArray;

/// A tensor resident on a compute backend. Shape metadata is
/// host-side and lazy-safe; `to_dense` is the explicit sync point
/// that forces the backend graph and widens to f64.
pub trait DeviceArray: std::fmt::Debug + Send + Sync {
    fn shape(&self) -> &[usize];
    fn to_dense(&self) -> DenseArray;
    fn as_any(&self) -> &dyn Any;
}

/// Shared pointer to a resident array. Backend ops return fresh
/// handles; clones are cheap graph references, never copies.
pub type Dev = Arc<dyn DeviceArray>;

/// Elementwise / contraction binaries (broadcast semantics are the
/// backend's, which for MLX matches the CPU path's scalar+array
/// and equal-shape cases the tape emits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
}

/// Elementwise unaries plus the rank-2 transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryKind {
    Neg,
    Exp,
    Log,
    Tanh,
    Sigmoid,
    Relu,
    Transpose,
}

/// Axis-aware ops. `axis = None` means "over all elements" for the
/// reductions and is invalid for the softmaxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisKind {
    Softmax,
    LogSoftmax,
    Sum,
    Mean,
}

/// Errors on the handle seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleError {
    /// No backend registered this process (CPU-only build or the
    /// binary never called [`crate::register_device_ops`]).
    NoBackend,
    /// Both operands are host-resident; use the `DenseArray` path.
    NotResident,
    /// The backend rejected the op (shape, dtype, downcast, ...).
    Backend(String),
}

impl std::fmt::Display for HandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBackend => write!(f, "no device backend registered in this process"),
            Self::NotResident => write!(f, "operands are host-resident; use the CPU path"),
            Self::Backend(msg) => write!(f, "device backend error: {msg}"),
        }
    }
}

impl std::error::Error for HandleError {}
