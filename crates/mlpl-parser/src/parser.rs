//! Parser: transforms token stream into AST.

use mlpl_core::Span;

use crate::ast::Expr;
use crate::error::ParseError;
use crate::token::{Token, TokenKind};

/// Parse a token stream into a list of expression statements.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let mut p = Parser::new(tokens);
    let mut stmts = Vec::new();
    p.skip_sep();
    while p.pos < p.tokens.len() && p.tokens[p.pos].kind != TokenKind::Eof {
        stmts.push(p.parse_expr()?);
        p.skip_sep();
    }
    Ok(stmts)
}

pub(crate) struct Parser<'a> {
    pub(crate) tokens: &'a [Token],
    pub(crate) pos: usize,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse a single expression.
    /// Steps 003-004 will extend this with binop, assignment, and fn calls.
    pub(crate) fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let tok = &self.tokens[self.pos];
        match &tok.kind {
            TokenKind::IntLit(n) => {
                let expr = Expr::IntLit(*n, tok.span);
                self.pos += 1;
                Ok(expr)
            }
            TokenKind::FloatLit(f) => {
                let expr = Expr::FloatLit(*f, tok.span);
                self.pos += 1;
                Ok(expr)
            }
            TokenKind::Ident(name) => {
                let expr = Expr::Ident(name.clone(), tok.span);
                self.pos += 1;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_array_lit(),
            TokenKind::LParen => self.parse_paren(),
            _ => Err(ParseError::UnexpectedToken {
                found: format!("{:?}", tok.kind),
                span: tok.span,
            }),
        }
    }

    fn parse_array_lit(&mut self) -> Result<Expr, ParseError> {
        let open_span = self.tokens[self.pos].span;
        self.pos += 1;
        let mut elems = Vec::new();
        if !self.is(TokenKind::RBracket) {
            elems.push(self.parse_expr()?);
            while self.is(TokenKind::Comma) {
                self.pos += 1;
                elems.push(self.parse_expr()?);
            }
        }
        if !self.is(TokenKind::RBracket) {
            return Err(ParseError::UnclosedDelimiter {
                open: "[".into(),
                span: open_span,
            });
        }
        let close_span = self.tokens[self.pos].span;
        self.pos += 1;
        Ok(Expr::ArrayLit(
            elems,
            Span::new(open_span.start, close_span.end),
        ))
    }

    fn parse_paren(&mut self) -> Result<Expr, ParseError> {
        let open_span = self.tokens[self.pos].span;
        self.pos += 1;
        let expr = self.parse_expr()?;
        if !self.is(TokenKind::RParen) {
            return Err(ParseError::UnclosedDelimiter {
                open: "(".into(),
                span: open_span,
            });
        }
        self.pos += 1;
        Ok(expr)
    }

    pub(crate) fn is(&self, kind: TokenKind) -> bool {
        self.pos < self.tokens.len()
            && std::mem::discriminant(&self.tokens[self.pos].kind) == std::mem::discriminant(&kind)
    }

    pub(crate) fn skip_sep(&mut self) {
        while self.pos < self.tokens.len()
            && matches!(
                self.tokens[self.pos].kind,
                TokenKind::Newline | TokenKind::Semicolon
            )
        {
            self.pos += 1;
        }
    }
}
