//! Statement-shaped parser methods split out of `parser.rs` so that
//! module stays under the sw-checklist function-count budget.
//!
//! Every method here takes `&mut Parser` as the receiver via an
//! inherent-impl extension block. Saga 12 step 003 extraction
//! point.

use mlpl_core::Span;

use crate::ast::Expr;
use crate::error::{ParseError, describe_kind};
use crate::parser::Parser;
use crate::token::TokenKind;

impl Parser<'_> {
    /// Parse `name : [axis1, axis2, ...] = value` and desugar to
    /// `Assign { name, value: FnCall("label", [value, ArrayLit([StrLit(a1), ...])]) }`.
    /// Saga 11.5 Phase 2. An empty list `[]` is legal and means
    /// "scalar with no axes".
    pub(crate) fn parse_annotated_assign(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        let TokenKind::Ident(n) = &self.tokens[self.pos].kind else {
            unreachable!()
        };
        let name = n.clone();
        self.pos += 2; // skip name and ':'
        let br_start = self.expect(&TokenKind::LBracket)?;
        let mut labels: Vec<Expr> = Vec::new();
        while !self.is(TokenKind::RBracket) {
            let tok = &self.tokens[self.pos];
            let TokenKind::Ident(a) = &tok.kind else {
                return Err(ParseError::UnexpectedToken {
                    found: describe_kind(&tok.kind),
                    span: tok.span,
                });
            };
            labels.push(Expr::StrLit(a.clone(), tok.span));
            self.pos += 1;
            if self.is(TokenKind::Comma) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let br_end = self.expect(&TokenKind::RBracket)?;
        self.expect(&TokenKind::Equals)?;
        let value = self.parse_expr(0)?;
        let value_span = value.span();
        let call = Expr::FnCall {
            name: "label".into(),
            args: vec![
                value,
                Expr::ArrayLit(labels, Span::new(br_start.start, br_end.end)),
            ],
            span: Span::new(br_start.start, value_span.end),
        };
        Ok(Expr::Assign {
            name,
            value: Box::new(call),
            span: Span::new(start.start, value_span.end),
        })
    }

    /// Parse `experiment "name" { body }`. Saga 12 step 007.
    /// Name must be a string literal; anything else is a parse
    /// error to keep the grammar unambiguous.
    pub(crate) fn parse_experiment(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1; // skip 'experiment'
        let tok = &self.tokens[self.pos];
        let TokenKind::StrLit(name) = &tok.kind else {
            return Err(ParseError::UnexpectedToken {
                found: describe_kind(&tok.kind),
                span: tok.span,
            });
        };
        let name = name.clone();
        self.pos += 1;
        let (body, end) = self.parse_braced_body()?;
        Ok(Expr::Experiment {
            name,
            body,
            span: Span::new(start.start, end.end),
        })
    }

    /// Parse `for <ident> in <expr> { body }`. Saga 12 step 003.
    pub(crate) fn parse_for(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1; // skip 'for'
        let tok = &self.tokens[self.pos];
        let TokenKind::Ident(binding) = &tok.kind else {
            return Err(ParseError::UnexpectedToken {
                found: describe_kind(&tok.kind),
                span: tok.span,
            });
        };
        let binding = binding.clone();
        self.pos += 1;
        self.expect(&TokenKind::In)?;
        let source = self.parse_expr(0)?;
        let (body, end) = self.parse_braced_body()?;
        Ok(Expr::For {
            binding,
            source: Box::new(source),
            body,
            span: Span::new(start.start, end.end),
        })
    }

    /// Consume `{ stmt? (sep stmt)* }` and return the parsed body
    /// plus the closing brace's span. Shared by `parse_repeat`
    /// (which also handles `train`) and `parse_for`.
    pub(crate) fn parse_braced_body(&mut self) -> Result<(Vec<Expr>, Span), ParseError> {
        let open = self.expect(&TokenKind::LBrace)?;
        self.skip_sep();
        let mut body = Vec::new();
        while self.pos < self.tokens.len()
            && !self.is(TokenKind::RBrace)
            && self.tokens[self.pos].kind != TokenKind::Eof
        {
            body.push(self.parse_statement()?);
            self.skip_sep();
        }
        if !self.is(TokenKind::RBrace) {
            return Err(ParseError::UnclosedDelimiter {
                open: "{".into(),
                span: open,
            });
        }
        let end = self.tokens[self.pos].span;
        self.pos += 1;
        Ok((body, end))
    }
}
