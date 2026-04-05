//! Lexer for MLPL source code.

use mlpl_core::Span;

use crate::error::ParseError;
use crate::lex_util::{lex_number, single_char_token};
use crate::token::{Token, TokenKind};

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

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.bytes.len() {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span::new(self.pos, self.pos),
            });
        }
        let b = self.bytes[self.pos];
        if b == b'\n' {
            let tok = Token {
                kind: TokenKind::Newline,
                span: Span::new(self.pos, self.pos + 1),
            };
            self.pos += 1;
            self.prev_was_value = false;
            return Ok(tok);
        }
        if let Some(kind) = single_char_token(b) {
            let is_val = matches!(kind, TokenKind::RParen | TokenKind::RBracket);
            let tok = Token {
                kind,
                span: Span::new(self.pos, self.pos + 1),
            };
            self.pos += 1;
            self.prev_was_value = is_val;
            return Ok(tok);
        }
        if b == b'-' {
            return self.lex_minus();
        }
        if b.is_ascii_digit() {
            return self.lex_digit();
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            return Ok(self.lex_ident());
        }
        let ch = self.source[self.pos..].chars().next().unwrap();
        Err(ParseError::UnexpectedCharacter {
            ch,
            span: Span::new(self.pos, self.pos + ch.len_utf8()),
        })
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b' ' || b == b'\t' {
                self.pos += 1;
            } else if b == b'#' {
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn lex_minus(&mut self) -> Result<Token, ParseError> {
        if let Some((tok, end)) = lex_number(self.bytes, self.pos).filter(|_| !self.prev_was_value)
        {
            let span = Span::new(self.pos, end);
            self.pos = end;
            self.prev_was_value = true;
            return Ok(Token { kind: tok, span });
        }
        let tok = Token {
            kind: TokenKind::Minus,
            span: Span::new(self.pos, self.pos + 1),
        };
        self.pos += 1;
        self.prev_was_value = false;
        Ok(tok)
    }

    fn lex_digit(&mut self) -> Result<Token, ParseError> {
        if let Some((tok, end)) = lex_number(self.bytes, self.pos) {
            let span = Span::new(self.pos, end);
            self.pos = end;
            self.prev_was_value = true;
            return Ok(Token { kind: tok, span });
        }
        Err(ParseError::InvalidNumber {
            span: Span::new(self.pos, self.pos + 1),
        })
    }

    fn lex_ident(&mut self) -> Token {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        let name = std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap()
            .to_owned();
        self.prev_was_value = true;
        Token {
            kind: TokenKind::Ident(name),
            span: Span::new(start, self.pos),
        }
    }
}
