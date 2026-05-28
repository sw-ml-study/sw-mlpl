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
    /// `.` (field access on records, Saga 29 step 001). Distinct
    /// from float-literal decimal points -- floats lex via
    /// `lex_number` which only consumes `digit.digit`; a bare `.`
    /// at the lexer fall-through is this token.
    Dot,
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
    /// `if` keyword (Saga 31 step 004). Introduces an `if cond
    /// { then } else { else }` expression that returns the value
    /// of whichever branch was taken. `else` is required.
    If,
    /// `else` keyword (Saga 31 step 004). Always paired with `if`.
    Else,
    /// `while` keyword (Saga 31 step 005). Introduces a `while cond
    /// { body }` loop. Exits when `cond` is falsy or `break` fires.
    While,
    /// `break` keyword (Saga 31 step 005). Exits the nearest
    /// enclosing `while` loop. Optionally followed by a value.
    Break,
    /// `continue` keyword (Saga 31 step 005). Skips the rest of
    /// the current `while` iteration; cond is re-checked.
    Continue,
    /// `def` keyword (Saga 46). Introduces a user-defined function.
    Def,
    /// `return` keyword (Saga 46). Early exit from a UDF body.
    Return,
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
