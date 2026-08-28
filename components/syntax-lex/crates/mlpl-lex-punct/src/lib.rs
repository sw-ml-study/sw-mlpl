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
        e = extend_user_ref(bytes, s, e);
        let name = std::str::from_utf8(&bytes[s..e]).unwrap().to_owned();
        Some((TokenKind::BuiltinRef(name), e))
    } else if matches!(next, b'+' | b'*' | b'/' | b'-') {
        Some((TokenKind::BuiltinRef((next as char).to_string()), pos + 2))
    } else {
        None
    }
}

/// Match a comparison/assignment operator, longest-match first, so the
/// two-char forms (`<=`, `>=`, `==`, `!=`) win over the one-char forms
/// (`<`, `>`, `=`). `!` alone is not an operator yet (unary `not` is a
/// later rung), so it falls through to an unexpected-character error.
/// Runs BEFORE `single_char_token` so `==` never lexes as two `=`.
pub fn lex_operator(bytes: &[u8], pos: usize) -> Option<(TokenKind, usize)> {
    let b = *bytes.get(pos)?;
    let next = bytes.get(pos + 1).copied();
    match (b, next) {
        (b'<', Some(b'=')) => Some((TokenKind::Le, pos + 2)),
        (b'>', Some(b'=')) => Some((TokenKind::Ge, pos + 2)),
        (b'=', Some(b'=')) => Some((TokenKind::EqEq, pos + 2)),
        (b'!', Some(b'=')) => Some((TokenKind::Ne, pos + 2)),
        (b'<', _) => Some((TokenKind::Lt, pos + 1)),
        (b'>', _) => Some((TokenKind::Gt, pos + 1)),
        (b'=', _) => Some((TokenKind::Equals, pos + 1)),
        _ => None,
    }
}

/// Match a single-char punctuation/operator token (except minus and
/// the comparison/assignment operators handled by `lex_operator`).
pub fn single_char_token(b: u8) -> Option<TokenKind> {
    match b {
        b'(' => Some(TokenKind::LParen),
        b')' => Some(TokenKind::RParen),
        b'[' => Some(TokenKind::LBracket),
        b']' => Some(TokenKind::RBracket),
        b',' => Some(TokenKind::Comma),
        b':' => Some(TokenKind::Colon),
        b';' => Some(TokenKind::Semicolon),
        b'@' => Some(TokenKind::At),
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

/// `:namespace:name` -- a quoted QUALIFIED reference (user function
/// `:u:area`, library `:result:zip`, extension `:native3d:camera`):
/// extend the token through the second identifier so the reference is
/// ONE `BuiltinRef` token holding `"namespace:name"`. A bare `:name`
/// (no second colon) stays a plain single-segment ref (a builtin like
/// `:max` or `:+`). Any namespace segment is accepted -- `u:` is no
/// longer special.
fn extend_user_ref(bytes: &[u8], _s: usize, mut e: usize) -> usize {
    if bytes.get(e) == Some(&b':')
        && bytes
            .get(e + 1)
            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
    {
        e += 1;
        while e < bytes.len() && (bytes[e].is_ascii_alphanumeric() || bytes[e] == b'_') {
            e += 1;
        }
    }
    e
}
