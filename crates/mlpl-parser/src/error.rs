//! Error types for mlpl-parser.

use mlpl_core::Span;

use crate::token::TokenKind;

/// Render a `TokenKind` as the human-readable string used in
/// `ParseError::UnexpectedToken` messages. Lives in this module
/// because it is part of error surface formatting.
pub(crate) fn describe_kind(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Eof => "end of input".into(),
        TokenKind::Newline => "newline".into(),
        TokenKind::IntLit(n) => format!("integer {n}"),
        TokenKind::FloatLit(n) => format!("float {n}"),
        TokenKind::StrLit(s) => format!("string \"{s}\""),
        TokenKind::Ident(s) => format!("identifier '{s}'"),
        TokenKind::LParen => "'('".into(),
        TokenKind::RParen => "')'".into(),
        TokenKind::LBracket => "'['".into(),
        TokenKind::RBracket => "']'".into(),
        TokenKind::LBrace => "'{'".into(),
        TokenKind::RBrace => "'}'".into(),
        TokenKind::Comma => "','".into(),
        TokenKind::Equals => "'='".into(),
        TokenKind::Colon => "':'".into(),
        TokenKind::Semicolon => "';'".into(),
        TokenKind::Plus => "'+'".into(),
        TokenKind::Minus => "'-'".into(),
        TokenKind::Star => "'*'".into(),
        TokenKind::Slash => "'/'".into(),
        TokenKind::Repeat => "'repeat'".into(),
        TokenKind::Train => "'train'".into(),
        TokenKind::For => "'for'".into(),
        TokenKind::In => "'in'".into(),
    }
}

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
    /// Bytes inside a string literal were not valid UTF-8. Saga 12.
    InvalidUtf8 {
        /// Byte span of the offending sequence.
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
            Self::InvalidUtf8 { span } => {
                write!(f, "invalid UTF-8 in string literal at {span}")
            }
        }
    }
}

impl std::error::Error for ParseError {}
