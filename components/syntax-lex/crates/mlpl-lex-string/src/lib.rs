//! String-literal lexing (double-quoted, escapes, UTF-8 aware).

use mlpl_core::Span;
use mlpl_lexer_error::ParseError;
use mlpl_lexer_token::TokenKind;

/// Lex a double-quoted string literal starting at `start`.
/// Returns the token and the byte offset just after the closing quote.
pub fn lex_string(bytes: &[u8], start: usize) -> Result<(TokenKind, usize), ParseError> {
    let mut pos = start + 1;
    let mut value = String::new();
    loop {
        if pos >= bytes.len() {
            return Err(unclosed(start));
        }
        match bytes[pos] {
            b'"' => return Ok((TokenKind::StrLit(value), pos + 1)),
            b'\\' => pos = handle_escape(bytes, pos, &mut value, start)?,
            _ => pos = handle_utf8_char(bytes, pos, &mut value)?,
        }
    }
}

fn unclosed(start: usize) -> ParseError {
    ParseError::UnclosedDelimiter {
        open: "\"".into(),
        span: Span::new(start, start + 1),
    }
}

fn handle_escape(
    bytes: &[u8],
    pos: usize,
    value: &mut String,
    start: usize,
) -> Result<usize, ParseError> {
    let pos = pos + 1;
    if pos >= bytes.len() {
        return Err(unclosed(start));
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
    Ok(pos + 1)
}

fn handle_utf8_char(bytes: &[u8], pos: usize, value: &mut String) -> Result<usize, ParseError> {
    let n = utf8_char_len(bytes[pos]).ok_or(ParseError::InvalidUtf8 {
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
    Ok(end)
}

fn utf8_char_len(b: u8) -> Option<usize> {
    match b {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}
