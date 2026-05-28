//! Identifier + keyword lexing.

use mlpl_lexer_token::TokenKind;

/// Lex an identifier (possibly a keyword) starting at `start`.
/// Returns the resolved `TokenKind` and the end byte offset.
pub fn lex_ident(bytes: &[u8], start: usize) -> (TokenKind, usize) {
    let mut e = start;
    while e < bytes.len() && (bytes[e].is_ascii_alphanumeric() || bytes[e] == b'_') {
        e += 1;
    }
    let name = std::str::from_utf8(&bytes[start..e]).unwrap().to_owned();
    (classify(&name), e)
}

fn classify(name: &str) -> TokenKind {
    match name {
        "repeat" => TokenKind::Repeat,
        "train" => TokenKind::Train,
        "for" => TokenKind::For,
        "in" => TokenKind::In,
        "experiment" => TokenKind::Experiment,
        "device" => TokenKind::Device,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "while" => TokenKind::While,
        "break" => TokenKind::Break,
        "continue" => TokenKind::Continue,
        "def" => TokenKind::Def,
        "return" => TokenKind::Return,
        _ => TokenKind::Ident(name.to_owned()),
    }
}
