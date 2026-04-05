//! Runtime error types.

/// Errors produced by built-in function dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeError {
    /// Function name not recognized.
    UnknownFunction(String),
    /// Wrong number of arguments.
    ArityMismatch {
        /// Function name.
        func: String,
        /// Expected argument count.
        expected: usize,
        /// Actual argument count.
        got: usize,
    },
    /// Argument failed a precondition.
    InvalidArgument {
        /// Function name.
        func: String,
        /// What went wrong.
        reason: String,
    },
    /// Propagated array error.
    ArrayError(mlpl_array::ArrayError),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFunction(name) => write!(f, "unknown function: {name}"),
            Self::ArityMismatch {
                func,
                expected,
                got,
            } => write!(f, "{func}: expected {expected} args, got {got}"),
            Self::InvalidArgument { func, reason } => {
                write!(f, "{func}: {reason}")
            }
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<mlpl_array::ArrayError> for RuntimeError {
    fn from(e: mlpl_array::ArrayError) -> Self {
        Self::ArrayError(e)
    }
}
