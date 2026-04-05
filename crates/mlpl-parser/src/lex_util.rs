//! Lexer utility functions.

use crate::token::TokenKind;

/// Try to lex a number (integer or float) starting at `pos`.
pub(crate) fn lex_number(bytes: &[u8], start: usize) -> Option<(TokenKind, usize)> {
    let mut pos = start;
    if pos < bytes.len() && bytes[pos] == b'-' {
        pos += 1;
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_digit() {
        return None;
    }
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos < bytes.len()
        && bytes[pos] == b'.'
        && pos + 1 < bytes.len()
        && bytes[pos + 1].is_ascii_digit()
    {
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        let val: f64 = std::str::from_utf8(&bytes[start..pos])
            .unwrap()
            .parse()
            .ok()?;
        Some((TokenKind::FloatLit(val), pos))
    } else {
        let val: i64 = std::str::from_utf8(&bytes[start..pos])
            .unwrap()
            .parse()
            .ok()?;
        Some((TokenKind::IntLit(val), pos))
    }
}

/// Match a single-char punctuation/operator token (except minus).
pub(crate) fn single_char_token(b: u8) -> Option<TokenKind> {
    match b {
        b'(' => Some(TokenKind::LParen),
        b')' => Some(TokenKind::RParen),
        b'[' => Some(TokenKind::LBracket),
        b']' => Some(TokenKind::RBracket),
        b',' => Some(TokenKind::Comma),
        b'=' => Some(TokenKind::Equals),
        b';' => Some(TokenKind::Semicolon),
        b'+' => Some(TokenKind::Plus),
        b'*' => Some(TokenKind::Star),
        b'/' => Some(TokenKind::Slash),
        _ => None,
    }
}
