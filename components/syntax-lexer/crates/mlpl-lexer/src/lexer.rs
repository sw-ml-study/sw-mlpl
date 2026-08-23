//! `Lexer` driver: walks source bytes, delegates to per-concern
//! sibling crates, and produces a flat `Vec<Token>`.

use mlpl_core::Span;
use mlpl_lex_ident::lex_ident;
use mlpl_lex_number::lex_number;
use mlpl_lex_punct::{lex_builtin_ref, lex_operator, single_char_token, skip_whitespace};
use mlpl_lex_string::lex_string;
use mlpl_lexer_error::ParseError;
use mlpl_lexer_token::{Token, TokenKind};

/// Tokenize MLPL source code.
pub fn lex(source: &str) -> Result<Vec<Token>, ParseError> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        let tok = lexer.next_token()?;
        let done = tok.kind == TokenKind::Eof;
        tokens.push(tok);
        if done {
            break;
        }
    }
    Ok(tokens)
}

struct Lexer<'a> {
    bytes: &'a [u8],
    source: &'a str,
    pos: usize,
    prev_was_value: bool,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            bytes: source.as_bytes(),
            source,
            pos: 0,
            prev_was_value: false,
        }
    }

    fn make_tok(&mut self, kind: TokenKind, end: usize, prev_was_value: bool) -> Token {
        let span = Span::new(self.pos, end);
        self.pos = end;
        self.prev_was_value = prev_was_value;
        Token { kind, span }
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.pos = skip_whitespace(self.bytes, self.pos);
        if self.pos >= self.bytes.len() {
            return Ok(self.make_tok(TokenKind::Eof, self.pos, false));
        }
        let b = self.bytes[self.pos];
        if b == b'\n' {
            return Ok(self.make_tok(TokenKind::Newline, self.pos + 1, false));
        }
        if let Some((kind, end)) = lex_builtin_ref(self.bytes, self.pos) {
            return Ok(self.make_tok(kind, end, true));
        }
        if let Some((kind, end)) = lex_operator(self.bytes, self.pos) {
            return Ok(self.make_tok(kind, end, false));
        }
        if let Some(kind) = single_char_token(b) {
            let is_val = matches!(kind, TokenKind::RParen | TokenKind::RBracket);
            return Ok(self.make_tok(kind, self.pos + 1, is_val));
        }
        if b == b'"' {
            let (kind, end) = lex_string(self.bytes, self.pos)?;
            return Ok(self.make_tok(kind, end, true));
        }
        if b == b'-' {
            return self.lex_minus();
        }
        if b.is_ascii_digit() {
            return self.lex_digit();
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let (kind, end) = lex_ident(self.bytes, self.pos);
            let is_val = matches!(kind, TokenKind::Ident(_));
            return Ok(self.make_tok(kind, end, is_val));
        }
        let ch = self.source[self.pos..].chars().next().unwrap();
        Err(ParseError::UnexpectedCharacter {
            ch,
            span: Span::new(self.pos, self.pos + ch.len_utf8()),
        })
    }

    fn lex_minus(&mut self) -> Result<Token, ParseError> {
        if let Some((kind, end)) = lex_number(self.bytes, self.pos).filter(|_| !self.prev_was_value)
        {
            return Ok(self.make_tok(kind, end, true));
        }
        Ok(self.make_tok(TokenKind::Minus, self.pos + 1, false))
    }

    fn lex_digit(&mut self) -> Result<Token, ParseError> {
        if let Some((kind, end)) = lex_number(self.bytes, self.pos) {
            return Ok(self.make_tok(kind, end, true));
        }
        Err(ParseError::InvalidNumber {
            span: Span::new(self.pos, self.pos + 1),
        })
    }
}
