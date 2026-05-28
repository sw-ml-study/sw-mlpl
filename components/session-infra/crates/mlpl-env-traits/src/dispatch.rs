//! `HasDispatch`: device-aware builtin dispatch. The first
//! capability trait that abstracts *behavior* rather than just
//! state access. Used by sub-crates that need to run primitive
//! ops (matmul, etc.) without knowing about mlpl-eval's
//! `crate::device::dispatched_call` (which routes through MLX
//! when the active device is "mlx", CPU otherwise).
//!
//! Sub-crates take `&impl HasDispatch + HasOtherTraits`; the
//! consumer (mlpl-eval) impls it on `Environment` by delegating
//! to its existing dispatched_call helper.

use mlpl_array::{ArrayError, DenseArray};

/// Error vocabulary for `HasDispatch::dispatch`. Sub-crates
/// convert this to their own error type via `From`.
#[derive(Debug)]
pub enum DispatchError {
    UnknownOp(String),
    Runtime(String),
    ArrayError(ArrayError),
}

impl From<ArrayError> for DispatchError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOp(op) => write!(f, "unknown op '{op}'"),
            Self::Runtime(msg) => write!(f, "runtime: {msg}"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for DispatchError {}

pub trait HasDispatch {
    /// Run primitive op `op` with `args` against the active
    /// device. Implementations route through CPU or MLX (or
    /// future backends) as appropriate.
    fn dispatch(&self, op: &str, args: Vec<DenseArray>) -> Result<DenseArray, DispatchError>;
}
