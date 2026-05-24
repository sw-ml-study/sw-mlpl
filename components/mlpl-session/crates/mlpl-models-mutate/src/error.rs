//! Local error vocabulary for the model-mutate operations.

use mlpl_array::ArrayError;

#[derive(Debug)]
pub enum MutateError {
    BadArity {
        func: String,
        expected: usize,
        got: usize,
    },
    NotAModel {
        func: String,
        name: String,
    },
    NotAModelExpr(String),
    UnknownFamily {
        family: String,
        valid: Vec<String>,
    },
    UndefinedVariable(String),
    ExpectedString(String),
    ExpectedScalar(String),
    ArrayError(ArrayError),
    RuntimeMessage(String),
}

impl From<ArrayError> for MutateError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl std::fmt::Display for MutateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadArity {
                func,
                expected,
                got,
            } => write!(f, "{func} expects {expected} arguments, got {got}"),
            Self::NotAModel { func, name } => write!(f, "{func}: '{name}' is not a model"),
            Self::NotAModelExpr(func) => {
                write!(f, "{func}: argument must evaluate to a model")
            }
            Self::UnknownFamily { family, valid } => write!(
                f,
                "unknown family '{family}' (expected one of {})",
                valid.join(", ")
            ),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::ExpectedString(func) => {
                write!(f, "{func}: argument must be a string literal")
            }
            Self::ExpectedScalar(func) => write!(f, "{func}: expected a scalar"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
            Self::RuntimeMessage(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MutateError {}
