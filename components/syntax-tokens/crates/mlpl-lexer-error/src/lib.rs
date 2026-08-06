//! Error types for MLPL lexing and parsing.

use mlpl_core::Span;

use mlpl_lexer_token::TokenKind;

/// Render a `TokenKind` as the human-readable string used in
/// `ParseError::UnexpectedToken` messages. Lives in this module
/// because it is part of error surface formatting.
pub fn describe_kind(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Eof => "end of input".into(),
        TokenKind::Newline => "newline".into(),
        TokenKind::At => "'@'".into(),
        TokenKind::IntLit(n) => format!("integer {n}"),
        TokenKind::FloatLit(n) => format!("float {n}"),
        TokenKind::StrLit(s) => format!("string \"{s}\""),
        TokenKind::Ident(s) => format!("identifier '{s}'"),
        TokenKind::BuiltinRef(s) => format!("builtin ref ':{s}'"),
        TokenKind::LParen => "'('".into(),
        TokenKind::RParen => "')'".into(),
        TokenKind::LBracket => "'['".into(),
        TokenKind::RBracket => "']'".into(),
        TokenKind::LBrace => "'{'".into(),
        TokenKind::RBrace => "'}'".into(),
        TokenKind::Dot => "'.'".into(),
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
        TokenKind::Experiment => "'experiment'".into(),
        TokenKind::Device => "'device'".into(),
        TokenKind::If => "'if'".into(),
        TokenKind::Else => "'else'".into(),
        TokenKind::While => "'while'".into(),
        TokenKind::Break => "'break'".into(),
        TokenKind::Continue => "'continue'".into(),
        TokenKind::Try => "'try'".into(),
        TokenKind::Catch => "'catch'".into(),
        TokenKind::Question => "'?'".into(),
        TokenKind::Def => "'def'".into(),
        TokenKind::Return => "'return'".into(),
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
    /// A record literal repeated a field name (Saga 29 step 001).
    /// `{X: 1, X: 2}` errors here rather than silently picking one,
    /// so the eval path can assume field names are unique.
    DuplicateRecordField {
        /// The repeated field name.
        name: String,
        /// Span of the second occurrence.
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
            Self::DuplicateRecordField { name, span } => {
                write!(f, "duplicate record field '{name}' at {span}")
            }
        }
    }
}

impl std::error::Error for ParseError {}
