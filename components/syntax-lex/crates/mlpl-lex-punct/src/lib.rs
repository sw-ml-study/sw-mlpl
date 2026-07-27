//! Punctuation + whitespace + builtin-ref helpers for the lexer.

use mlpl_lexer_token::TokenKind;

/// Skip ASCII spaces, tabs, and `#` line comments. Newlines stop the
/// skip -- they are statement separators in MLPL.
pub fn skip_whitespace(bytes: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < bytes.len() {
        match bytes[p] {
            b' ' | b'\t' => p += 1,
            b'#' => {
                while p < bytes.len() && bytes[p] != b'\n' {
                    p += 1;
                }
            }
            _ => break,
        }
    }
    p
}

/// Try to lex a `:foo` / `:+` / `:max` `BuiltinRef` starting at `pos`.
pub fn lex_builtin_ref(bytes: &[u8], pos: usize) -> Option<(TokenKind, usize)> {
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
pub fn single_char_token(b: u8) -> Option<TokenKind> {
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
        b'?' => Some(TokenKind::Question),
        _ => None,
    }
}
