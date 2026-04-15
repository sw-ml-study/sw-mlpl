//! Lexer utility functions.

use mlpl_core::Span;

use crate::error::ParseError;
use crate::token::TokenKind;

/// Lex a double-quoted string literal starting at `start` (which points
/// at the opening `"`). Returns the token and the byte offset just after
/// the closing quote.
pub(crate) fn lex_string(bytes: &[u8], start: usize) -> Result<(TokenKind, usize), ParseError> {
    let mut pos = start + 1; // skip opening quote
    let mut value = String::new();
    loop {
        if pos >= bytes.len() {
            return Err(ParseError::UnclosedDelimiter {
                open: "\"".into(),
                span: Span::new(start, start + 1),
            });
        }
        let b = bytes[pos];
        if b == b'"' {
            return Ok((TokenKind::StrLit(value), pos + 1));
        }
        if b == b'\\' {
            pos += 1;
            if pos >= bytes.len() {
                return Err(ParseError::UnclosedDelimiter {
                    open: "\"".into(),
                    span: Span::new(start, start + 1),
                });
            }
            let ch = match bytes[pos] {
                b'"' => '"',
                b'\\' => '\\',
                b'n' => '\n',
                b't' => '\t',
                b'r' => '\r',
                other => {
                    return Err(ParseError::UnexpectedCharacter {
                        ch: other as char,
                        span: Span::new(pos - 1, pos + 1),
                    });
                }
            };
            value.push(ch);
            pos += 1;
        } else {
            value.push(b as char);
            pos += 1;
        }
    }
}

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
        b':' => Some(TokenKind::Colon),
        b';' => Some(TokenKind::Semicolon),
        b'+' => Some(TokenKind::Plus),
        b'*' => Some(TokenKind::Star),
        b'/' => Some(TokenKind::Slash),
        b'{' => Some(TokenKind::LBrace),
        b'}' => Some(TokenKind::RBrace),
        _ => None,
    }
}
