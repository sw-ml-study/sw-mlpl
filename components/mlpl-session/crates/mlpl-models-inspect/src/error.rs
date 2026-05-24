//! Local error vocabulary for the inspect crate.

use mlpl_array::ArrayError;

#[derive(Debug)]
pub enum InspectError {
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
    NoEmbedding,
    NoTrainableParams,
    NotAScalar {
        func: String,
        name: String,
        rank: usize,
    },
    NotPositive {
        func: String,
        name: String,
        value: f64,
    },
    ArrayError(ArrayError),
}

impl From<ArrayError> for InspectError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl std::fmt::Display for InspectError {
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
            Self::NoEmbedding => write!(f, "embed_table: model contains no Embedding layer"),
            Self::NoTrainableParams => {
                write!(f, "estimate_train: model has no trainable parameters")
            }
            Self::NotAScalar { func, name, rank } => {
                write!(f, "{func}: {name} must be a scalar, got rank {rank}")
            }
            Self::NotPositive { func, name, value } => {
                write!(f, "{func}: {name} must be positive, got {value}")
            }
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for InspectError {}
