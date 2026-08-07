//! Token-stream shaping ahead of parsing.

use mlpl_lexer::{Token, TokenKind};

/// Newlines are statement separators -- EXCEPT inside an open
/// `(` or `[`, where a statement plainly continues (an argument
/// list or vector literal spanning lines). Braces stay opaque:
/// `{` delimits blocks, where newlines separate statements.
pub(crate) fn continue_in_brackets(tokens: &[Token]) -> Vec<Token> {
    let mut depth = 0i64;
    let mut out = Vec::with_capacity(tokens.len());
    for t in tokens {
        match t.kind {
            TokenKind::LParen | TokenKind::LBracket => depth += 1,
            TokenKind::RParen | TokenKind::RBracket => depth = (depth - 1).max(0),
            TokenKind::Newline if depth > 0 => continue,
            _ => {}
        }
        out.push(t.clone());
    }
    out
}
