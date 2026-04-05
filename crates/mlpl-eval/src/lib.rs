//! Expression evaluator for MLPL.
//!
//! PoC: evaluates a sequence of numeric literals into a DenseArray.

use mlpl_array::DenseArray;
use mlpl_parser::{Token, TokenKind};

/// Errors produced during evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// No values to evaluate.
    EmptyInput,
    /// Token not supported in this PoC evaluator.
    Unsupported(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Evaluate a token stream into a DenseArray.
///
/// PoC: only handles sequences of numeric literals (IntLit, FloatLit).
/// A single number produces a scalar. Multiple numbers produce a vector.
pub fn evaluate(tokens: &[Token]) -> Result<DenseArray, EvalError> {
    let mut values = Vec::new();

    for tok in tokens {
        match &tok.kind {
            TokenKind::IntLit(n) => values.push(*n as f64),
            TokenKind::FloatLit(f) => values.push(*f),
            TokenKind::Eof | TokenKind::Newline => {}
            other => {
                return Err(EvalError::Unsupported(format!("{other:?}")));
            }
        }
    }

    if values.is_empty() {
        return Err(EvalError::EmptyInput);
    }

    if values.len() == 1 {
        Ok(DenseArray::from_scalar(values[0]))
    } else {
        Ok(DenseArray::from_vec(values))
    }
}
