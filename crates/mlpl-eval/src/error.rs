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
    /// Expected a string value but got something else.
    ExpectedString,
    /// Wrong number of arguments to a built-in.
    BadArity {
        /// Function name.
        func: String,
        /// Expected count.
        expected: usize,
        /// Got count.
        got: usize,
    },
    /// Error from the visualization layer.
    VizError(mlpl_viz::VizError),
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
            Self::ExpectedString => write!(f, "expected a string value"),
            Self::BadArity {
                func,
                expected,
                got,
            } => {
                write!(f, "{func} expects {expected} arguments, got {got}")
            }
            Self::VizError(e) => write!(f, "{e}"),
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

impl From<mlpl_viz::VizError> for EvalError {
    fn from(e: mlpl_viz::VizError) -> Self {
        Self::VizError(e)
    }
}
