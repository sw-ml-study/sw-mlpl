//! Parser: transforms token stream into AST.

use mlpl_core::Span;

use crate::ast::{BinOpKind, Expr};
use crate::error::ParseError;
use crate::token::{Token, TokenKind};

/// Parse a token stream into a list of expression statements.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let mut p = Parser::new(tokens);
    let mut stmts = Vec::new();
    p.skip_sep();
    while p.pos < p.tokens.len() && p.tokens[p.pos].kind != TokenKind::Eof {
        stmts.push(p.parse_expr(0)?);
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

    /// Parse an expression with precedence climbing (min_prec=0 for full expr).
    pub(crate) fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_atom()?;
        while let Some((op, prec)) = self.binop_prec() {
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let rhs = self.parse_expr(prec + 1)?;
            let span = Span::new(lhs.span().start, rhs.span().end);
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn binop_prec(&self) -> Option<(BinOpKind, u8)> {
        match self.tokens.get(self.pos).map(|t| &t.kind)? {
            TokenKind::Plus => Some((BinOpKind::Add, 1)),
            TokenKind::Minus => Some((BinOpKind::Sub, 1)),
            TokenKind::Star => Some((BinOpKind::Mul, 2)),
            TokenKind::Slash => Some((BinOpKind::Div, 2)),
            _ => None,
        }
    }

    pub(crate) fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let tok = &self.tokens[self.pos];
        match &tok.kind {
            TokenKind::IntLit(n) => {
                let e = Expr::IntLit(*n, tok.span);
                self.pos += 1;
                Ok(e)
            }
            TokenKind::FloatLit(f) => {
                let e = Expr::FloatLit(*f, tok.span);
                self.pos += 1;
                Ok(e)
            }
            TokenKind::Ident(name) => {
                let e = Expr::Ident(name.clone(), tok.span);
                self.pos += 1;
                Ok(e)
            }
            TokenKind::LBracket => self.parse_array_lit(),
            TokenKind::LParen => {
                let open = tok.span;
                self.pos += 1;
                let expr = self.parse_expr(0)?;
                if !self.is(TokenKind::RParen) {
                    return Err(ParseError::UnclosedDelimiter {
                        open: "(".into(),
                        span: open,
                    });
                }
                self.pos += 1;
                Ok(expr)
            }
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
            elems.push(self.parse_expr(0)?);
            while self.is(TokenKind::Comma) {
                self.pos += 1;
                elems.push(self.parse_expr(0)?);
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
