//! Error types for mlpl-parser.

use mlpl_core::Span;

/// Errors produced during parsing.
#[derive(Clone, Debug, PartialEq)]
pub enum ParseError {
    /// An unexpected character was encountered during lexing.
    UnexpectedCharacter {
        /// The character.
        ch: char,
        /// Where it was found.
        span: Span,
    },
    /// A number literal could not be parsed.
    InvalidNumber {
        /// Where the number was.
        span: Span,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedCharacter { ch, span } => {
                write!(f, "unexpected character '{ch}' at {span}")
            }
            Self::InvalidNumber { span } => {
                write!(f, "invalid number at {span}")
            }
        }
    }
}

impl std::error::Error for ParseError {}
