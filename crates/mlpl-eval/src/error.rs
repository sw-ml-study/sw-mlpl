//! Evaluation error types.

/// Errors produced during evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// No expressions to evaluate.
    EmptyInput,
    /// Variable not found in environment.
    UndefinedVariable(String),
    /// Feature not yet implemented.
    Unsupported(String),
    /// Error from array operations.
    ArrayError(mlpl_array::ArrayError),
    /// Repeat count must be a scalar.
    InvalidRepeatCount,
    /// Error from built-in function dispatch.
    RuntimeError(mlpl_runtime::RuntimeError),
    /// Expected an array value but got a string.
    ExpectedArray,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::InvalidRepeatCount => write!(f, "repeat count must be a scalar integer"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
            Self::RuntimeError(e) => write!(f, "{e}"),
            Self::ExpectedArray => write!(f, "expected an array value, got a string"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<mlpl_array::ArrayError> for EvalError {
    fn from(e: mlpl_array::ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl From<mlpl_runtime::RuntimeError> for EvalError {
    fn from(e: mlpl_runtime::RuntimeError) -> Self {
        Self::RuntimeError(e)
    }
}
