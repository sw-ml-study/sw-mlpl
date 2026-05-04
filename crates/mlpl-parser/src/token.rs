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
    /// Builtin / operator reference: `:foo`, `:+`, `:max`, `:*`.
    /// The lexer produces this whenever a `:` is immediately
    /// followed (no intervening space) by an identifier-start
    /// character or one of `+ * / -`. The annotation colon
    /// (`x : [batch] = ...`) keeps producing `Colon` because
    /// it requires a space after the `:`, then a `[`. This
    /// is the canonical way to pass a function / operator
    /// reference to a higher-order builtin like `reduce` or
    /// `map`. See `docs/glossary.md` "Reduce".
    BuiltinRef(String),
    /// String literal (double-quoted, escapes processed).
    StrLit(String),
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
    /// `:` (axis-label annotation, Saga 11.5 Phase 2).
    Colon,
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
    /// `train` keyword.
    Train,
    /// `for` keyword (streaming-iter, Saga 12 step 003).
    For,
    /// `in` keyword (only meaningful inside `for ... in ...`).
    In,
    /// `experiment` keyword (Saga 12 step 007).
    Experiment,
    /// `device` keyword (Saga 14 step 004). Introduces a scoped
    /// `device("mlx") { body }` or `device("cpu") { body }` block
    /// that dispatches ops inside the body through the named
    /// runtime target.
    Device,
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
