//! Local error vocabulary for `mlpl-models-tape`. The crate
//! does not depend on `mlpl-eval`'s rich `EvalError`; consumers
//! convert `TapeError` to whatever their own error type is
//! (`EvalError` in `mlpl-eval` does so via `From<TapeError>`).
//!
//! This is the C+D loose-coupling boundary: every sub-crate
//! has its own error vocabulary, and the upstream "controller"
//! is responsible for translation.

use mlpl_array::ArrayError;
use mlpl_core::LabeledShape;

#[derive(Debug)]
pub enum TapeError {
    UndefinedVariable(String),
    Unsupported(String),
    ShapeMismatch {
        op: String,
        expected: LabeledShape,
        actual: LabeledShape,
    },
    ArrayError(ArrayError),
}

impl From<ArrayError> for TapeError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl std::fmt::Display for TapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch in {op}: expected {expected:?}, got {actual:?}"
            ),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for TapeError {}
