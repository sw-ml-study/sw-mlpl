//! Error types for mlpl-parser.

use mlpl_core::Span;

/// Errors produced during lexing and parsing.
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
    /// An unexpected token was encountered during parsing.
    UnexpectedToken {
        /// Description of what was found.
        found: String,
        /// Where it was found.
        span: Span,
    },
    /// Expected a closing delimiter that was not found.
    UnclosedDelimiter {
        /// The opening delimiter.
        open: String,
        /// Where the opening delimiter was.
        span: Span,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedCharacter { ch, span } => {
                write!(f, "unexpected character '{ch}' at {span}")
            }
            Self::InvalidNumber { span } => write!(f, "invalid number at {span}"),
            Self::UnexpectedToken { found, span } => {
                write!(f, "unexpected token '{found}' at {span}")
            }
            Self::UnclosedDelimiter { open, span } => {
                write!(f, "unclosed '{open}' at {span}")
            }
        }
    }
}

impl std::error::Error for ParseError {}
