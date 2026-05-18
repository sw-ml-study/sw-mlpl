//! Lexer utility functions.

use mlpl_core::Span;

use crate::error::ParseError;
use crate::token::TokenKind;

/// Lex a double-quoted string literal starting at `start` (which points
/// at the opening `"`). Returns the token and the byte offset just after
/// the closing quote.
///
/// UTF-8 aware: non-ASCII characters inside the literal are decoded
/// via their full UTF-8 byte sequence and stored as proper Rust
/// `char`s. Malformed UTF-8 surfaces as `ParseError::InvalidUtf8`
/// with a byte-span pointing at the offending sequence.
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
            // Decode one UTF-8 code point from `bytes` starting at
            // `pos`. ASCII is a single byte; multi-byte leaders
            // advance by 2/3/4. `b as char` would be Latin-1 and
            // mojibake any non-ASCII input; Rust source is UTF-8 by
            // convention so we decode honestly.
            let n = utf8_char_len(b).ok_or(ParseError::InvalidUtf8 {
                span: Span::new(pos, pos + 1),
            })?;
            let end = pos + n;
            if end > bytes.len() {
                return Err(ParseError::InvalidUtf8 {
                    span: Span::new(pos, bytes.len()),
                });
            }
            let ch = std::str::from_utf8(&bytes[pos..end])
                .map_err(|_| ParseError::InvalidUtf8 {
                    span: Span::new(pos, end),
                })?
                .chars()
                .next()
                .expect("non-empty UTF-8 slice");
            value.push(ch);
            pos = end;
        }
    }
}

/// UTF-8 leading-byte length: 1 for ASCII, 2 for `110xxxxx`, 3 for
/// `1110xxxx`, 4 for `11110xxx`. Returns `None` for continuation
/// bytes (`10xxxxxx`) or invalid leaders -- treated as a UTF-8
/// decode failure.
fn utf8_char_len(b: u8) -> Option<usize> {
    match b {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
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

/// Skip ASCII spaces, tabs, and `#` line comments starting at
/// `pos`. Returns the new position. Newlines stop the skip --
/// they are statement separators in MLPL and produce `Newline`
/// tokens upstream.
pub(crate) fn skip_whitespace(bytes: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < bytes.len() {
        let b = bytes[p];
        if b == b' ' || b == b'\t' {
            p += 1;
        } else if b == b'#' {
            while p < bytes.len() && bytes[p] != b'\n' {
                p += 1;
            }
        } else {
            break;
        }
    }
    p
}

/// Try to lex a `:foo` / `:+` / `:max` BuiltinRef starting at
/// `pos`. Returns `(TokenKind::BuiltinRef(name), new_pos)` on
/// match, or `None` if `:` is followed by something other than
/// an ident-start char or one of `+ * / -` (so the caller can
/// fall through to the annotation `Colon` token).
pub(crate) fn lex_builtin_ref(bytes: &[u8], pos: usize) -> Option<(TokenKind, usize)> {
    if bytes.get(pos)? != &b':' {
        return None;
    }
    let next = *bytes.get(pos + 1)?;
    if next.is_ascii_alphabetic() || next == b'_' {
        let s = pos + 1;
        let mut e = s;
        while e < bytes.len() && (bytes[e].is_ascii_alphanumeric() || bytes[e] == b'_') {
            e += 1;
        }
        let name = std::str::from_utf8(&bytes[s..e]).unwrap().to_owned();
        Some((TokenKind::BuiltinRef(name), e))
    } else if matches!(next, b'+' | b'*' | b'/' | b'-') {
        Some((TokenKind::BuiltinRef((next as char).to_string()), pos + 2))
    } else {
        None
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
        b'.' => Some(TokenKind::Dot),
        _ => None,
    }
}
