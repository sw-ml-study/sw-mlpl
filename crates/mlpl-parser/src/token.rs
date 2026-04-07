//! Token types for the MLPL lexer.

use mlpl_core::Span;

/// The kind of a lexer token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    /// Integer literal.
    IntLit(i64),
    /// Float literal.
    FloatLit(f64),
    /// Identifier.
    Ident(String),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `=`
    Equals,
    /// `;`
    Semicolon,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `repeat` keyword.
    Repeat,
    /// Newline (statement separator).
    Newline,
    /// End of input.
    Eof,
}

/// A token with its source span.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// What kind of token.
    pub kind: TokenKind,
    /// Where in the source.
    pub span: Span,
}
