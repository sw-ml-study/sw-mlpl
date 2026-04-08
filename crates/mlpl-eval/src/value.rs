//! Runtime value type for MLPL.

use std::fmt;

use mlpl_array::DenseArray;

use crate::error::EvalError;
use crate::model::ModelSpec;

/// A runtime value: an array, a string, or a model.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A dense array (scalar, vector, matrix, ...).
    Array(DenseArray),
    /// A string (introduced for diagram type names, LLM prompts, etc.).
    Str(String),
    /// A model: a callable layer (or composition) with attached
    /// parameters held in the environment. Saga 11.
    Model(ModelSpec),
}

impl Value {
    /// Extract the inner array, returning an error if this is a
    /// string or model.
    pub fn into_array(self) -> Result<DenseArray, EvalError> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Str(_) | Self::Model(_) => Err(EvalError::ExpectedArray),
        }
    }

    /// Borrow the inner array, returning an error if this is a
    /// string or model.
    pub fn as_array(&self) -> Result<&DenseArray, EvalError> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Str(_) | Self::Model(_) => Err(EvalError::ExpectedArray),
        }
    }
}

impl From<DenseArray> for Value {
    fn from(a: DenseArray) -> Self {
        Self::Array(a)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Array(a) => write!(f, "{a}"),
            Self::Str(s) => write!(f, "{s}"),
            Self::Model(_) => write!(f, "<model>"),
        }
    }
}
