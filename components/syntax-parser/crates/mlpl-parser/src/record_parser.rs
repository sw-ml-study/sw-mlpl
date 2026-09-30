//! Record-literal and field-access parsing. Saga 29 step 001.
//!
//! Lives in a sibling module to keep `parser.rs` under the
//! sw-checklist function-count budget (same pattern as
//! `stmts.rs`). Two entry points hang off `Parser`:
//!
//! - `parse_record_lit_after_brace`: called from `parse_atom`
//!   when it sees `LBrace` in expression position. Consumes
//!   `{ field: expr (, field: expr)* }`. Empty record `{}` is
//!   legal. Trailing commas are accepted. Duplicate field names
//!   error at parse time so the eval path never has to choose.
//!
//! - `parse_postfix_chain`: called from `parse_expr` after
//!   `parse_atom` returns. Repeatedly consumes `?` and `.ident` and
//!   wraps the running expression in `FieldAccess`. Binds
//!   tighter than every infix binop (`f(x).y + z` parses as
//!   `(f(x).y) + z`).
//!
//! - `try_parse_destructure`: called from `parse_statement` at a
//!   statement-initial `{`; recognizes `{a, b: x} = value` and
//!   otherwise backtracks so the `{` parses as a record literal.
//!
//! Grammar disambiguation: `{` in expression position ALWAYS
//! opens a record literal because `{ stmt; ... }` blocks only
//! appear after the `repeat` / `train` / `for` / `experiment`
//! / `device` keywords -- those callers consume the `{`
//! directly via `parse_braced_body`, so `parse_atom` never
//! sees a `{` that should become a block.

use std::collections::HashSet;

use mlpl_core::Span;

use crate::parser::Parser;
use mlpl_lexer::TokenKind;

/// The identifier text a token contributes when it appears in a
/// MEMBER-NAME position (record-literal key or after `.`). Those
/// two spots are grammatically unambiguous, so keywords are plain
/// names there -- `s.train` and `{eval: 1}` are legal -- while
/// staying reserved everywhere else.
fn member_name(kind: &TokenKind) -> Option<String> {
    match kind {
        TokenKind::Ident(name) => Some(name.clone()),
        other => other.keyword().map(str::to_string),
    }
}
use mlpl_lexer::{ParseError, describe_kind};
use mlpl_parser_ast::Expr;

impl Parser<'_> {
    /// Caller has just consumed an `LBrace` at `open_span`.
    /// Parse fields up to the closing `RBrace`.
    pub(crate) fn parse_record_lit_after_brace(
        &mut self,
        open_span: Span,
    ) -> Result<Expr, ParseError> {
        let mut fields: Vec<(String, Expr)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        self.skip_sep();
        while !self.is(TokenKind::RBrace) {
            let name_tok = &self.tokens[self.pos];
            let Some(name) = member_name(&name_tok.kind) else {
                return Err(ParseError::UnexpectedToken {
                    found: describe_kind(&name_tok.kind),
                    span: name_tok.span,
                });
            };
            if !seen.insert(name.clone()) {
                return Err(ParseError::DuplicateRecordField {
                    name,
                    span: name_tok.span,
                });
            }
            self.pos += 1;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expr(0)?;
            fields.push((name, value));
            self.skip_sep();
            if self.is(TokenKind::Comma) {
                self.pos += 1;
                self.skip_sep();
            } else {
                break;
            }
        }
        if !self.is(TokenKind::RBrace) {
            return Err(ParseError::UnclosedDelimiter {
                open: "{".into(),
                span: open_span,
            });
        }
        let close_span = self.tokens[self.pos].span;
        self.pos += 1;
        Ok(Expr::RecordLit {
            fields,
            span: Span::new(open_span.start, close_span.end),
        })
    }

    /// At a statement-initial open brace: parse a destructuring
    /// assignment (`a, b: x` between the braces, then `= value`). Anything
    /// else -- a record literal, or a pattern with no `=` after it --
    /// restores the cursor and returns `None`, so the caller parses an
    /// expression as before.
    pub(crate) fn try_parse_destructure(&mut self) -> Result<Option<Expr>, ParseError> {
        let start = self.pos;
        self.pos += 1;
        let bindings: Vec<_> = std::iter::from_fn(|| self.pattern_binding()).collect();
        let closes = self.is(TokenKind::RBrace)
            && self.tokens.get(self.pos + 1).map(|t| &t.kind) == Some(&TokenKind::Equals);
        if !closes || bindings.is_empty() {
            self.pos = start;
            return Ok(None);
        }
        self.pos += 2;
        let value = self.parse_expr(0)?;
        let span = Span::new(self.tokens[start].span.start, value.span().end);
        Ok(Some(Expr::Destructure {
            bindings,
            value: Box::new(value),
            span,
        }))
    }

    /// One `field` or `field: var` pattern entry plus its separating comma
    /// (none before the close brace); `None` (cursor possibly advanced -- the caller
    /// restores it) when the tokens are not one.
    fn pattern_binding(&mut self) -> Option<(String, String)> {
        self.skip_newlines();
        let field = member_name(&self.tokens[self.pos].kind)?;
        self.pos += 1;
        let var = if self.is(TokenKind::Colon) {
            let TokenKind::Ident(var) = &self.tokens.get(self.pos + 1)?.kind else {
                return None;
            };
            self.pos += 2;
            var.clone()
        } else {
            field.clone()
        };
        self.skip_newlines();
        match self.tokens[self.pos].kind {
            TokenKind::Comma => self.pos += 1,
            TokenKind::RBrace => {}
            _ => return None,
        }
        Some((field, var))
    }

    /// Consume zero or more `.ident` postfix chains, wrapping
    /// `atom` in nested `FieldAccess` nodes.
    pub(crate) fn parse_postfix_chain(&mut self, mut atom: Expr) -> Result<Expr, ParseError> {
        // `expr?` -- Result propagation sugar (spike step 011):
        // desugars to `check(expr)` so no new AST node is needed.
        while self.is(TokenKind::Question) {
            let q_span = self.tokens[self.pos].span;
            self.pos += 1;
            let span = Span::new(atom.span().start, q_span.end);
            atom = Expr::FnCall {
                name: "check".into(),
                args: vec![atom],
                span,
            };
        }
        while self.is(TokenKind::Dot) {
            self.pos += 1;
            let name_tok = &self.tokens[self.pos];
            let Some(field) = member_name(&name_tok.kind) else {
                return Err(ParseError::UnexpectedToken {
                    found: describe_kind(&name_tok.kind),
                    span: name_tok.span,
                });
            };
            let field_span = name_tok.span;
            self.pos += 1;
            let span = Span::new(atom.span().start, field_span.end);
            atom = Expr::FieldAccess {
                receiver: Box::new(atom),
                field,
                span,
            };
        }
        Ok(atom)
    }
}
