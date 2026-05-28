//! Integer + float literal lexing.

use mlpl_lexer_token::TokenKind;

/// Try to lex a number (integer or float) starting at `pos`.
pub fn lex_number(bytes: &[u8], start: usize) -> Option<(TokenKind, usize)> {
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
    let has_fraction = pos < bytes.len()
        && bytes[pos] == b'.'
        && pos + 1 < bytes.len()
        && bytes[pos + 1].is_ascii_digit();
    if has_fraction {
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        let s = std::str::from_utf8(&bytes[start..pos]).unwrap();
        Some((TokenKind::FloatLit(s.parse().ok()?), pos))
    } else {
        let s = std::str::from_utf8(&bytes[start..pos]).unwrap();
        Some((TokenKind::IntLit(s.parse().ok()?), pos))
    }
}
