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
    }
    let s = std::str::from_utf8(&bytes[start..pos]).unwrap();
    finish(s, pos, has_fraction)
}

/// Parse the matched digits into a `Float`/`Int` token at `pos`.
fn finish(s: &str, pos: usize, is_float: bool) -> Option<(TokenKind, usize)> {
    let kind = if is_float {
        TokenKind::FloatLit(s.parse().ok()?)
    } else {
        TokenKind::IntLit(s.parse().ok()?)
    };
    Some((kind, pos))
}
