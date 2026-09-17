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
    let (pos, has_exp) = scan_exponent(bytes, pos);
    let s = std::str::from_utf8(&bytes[start..pos]).unwrap();
    finish(s, pos, has_fraction || has_exp)
}

/// Scan an optional `e`/`E` exponent (with optional `+`/`-` sign)
/// starting at `pos`. Returns the position past the exponent and whether
/// one was found. A bare `e` or `e-` with no following digit is NOT
/// consumed, so `1e` lexes as `1` then the identifier `e` (RS5).
fn scan_exponent(bytes: &[u8], pos: usize) -> (usize, bool) {
    if pos >= bytes.len() || (bytes[pos] != b'e' && bytes[pos] != b'E') {
        return (pos, false);
    }
    let mut ep = pos + 1;
    if ep < bytes.len() && (bytes[ep] == b'+' || bytes[ep] == b'-') {
        ep += 1;
    }
    if ep >= bytes.len() || !bytes[ep].is_ascii_digit() {
        return (pos, false);
    }
    while ep < bytes.len() && bytes[ep].is_ascii_digit() {
        ep += 1;
    }
    (ep, true)
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
