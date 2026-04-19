//! Evaluation error types.

use mlpl_core::LabeledShape;

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
    /// Tensor constructor shape dimension must be a non-negative scalar integer.
    InvalidShapeDim,
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
    /// Two operand shapes (or labels) disagree in a way the named op
    /// cannot resolve. Saga 11.5 Phase 4: replaces string
    /// `Unsupported` messages for broadcasting and contraction
    /// failures. `expected` and `actual` are the left- and right-hand
    /// operand labeled shapes respectively.
    ShapeMismatch {
        /// Operator or builtin name (`"add"`, `"matmul"`, ...).
        op: String,
        /// Left-hand operand's labeled shape.
        expected: LabeledShape,
        /// Right-hand operand's labeled shape.
        actual: LabeledShape,
    },
    /// Two tensors (or a tensor and the active `device("...") { }`
    /// scope) disagree on device placement. Saga 14 step 005: raised
    /// by `apply(model, X)` when the input lives on a different
    /// device than the model's parameters, and by any op that
    /// receives mixed-device operands. `op` names the site
    /// (`"matmul"`, `"apply"`, `"add"`, ...); `expected` is the
    /// device the left-hand side carries and `actual` is the
    /// right-hand side's.
    DeviceMismatch {
        /// Operator or builtin name.
        op: String,
        /// Device the left-hand (or first) operand is on.
        expected: String,
        /// Device the right-hand (or second) operand is on.
        actual: String,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::InvalidRepeatCount => write!(f, "repeat count must be a scalar integer"),
            Self::InvalidShapeDim => {
                write!(f, "shape dimension must be a non-negative scalar integer")
            }
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
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(f, "{op}: expected {expected}, got {actual}"),
            Self::DeviceMismatch {
                op,
                expected,
                actual,
            } => write!(f, "device mismatch: {op} on {expected} vs {actual}"),
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
