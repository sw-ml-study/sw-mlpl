//! Array-literal parsing (`[e, e, ...]`). Lives in a sibling module to
//! keep `parser.rs` under the sw-checklist file-LOC budget (same
//! pattern as `record_parser.rs` / `stmts.rs`). Both entry points hang
//! off `Parser` and are called from `parse_atom` when it sees `[`.

use mlpl_core::Span;
use mlpl_lexer::{ParseError, TokenKind};
use mlpl_parser_ast::Expr;

use crate::parser::Parser;

impl Parser<'_> {
    pub(crate) fn parse_array_lit(&mut self) -> Result<Expr, ParseError> {
        let open_span = self.tokens[self.pos].span;
        self.pos += 1;
        let elems = self.parse_array_elems()?;
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

    /// Parse the comma-separated elements of an array literal. Newlines
    /// are insignificant (commas separate), so a matrix can span lines.
    pub(crate) fn parse_array_elems(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut elems = Vec::new();
        self.skip_newlines();
        if !self.is(TokenKind::RBracket) {
            elems.push(self.parse_expr(0)?);
            self.skip_newlines();
            while self.is(TokenKind::Comma) {
                self.pos += 1;
                self.skip_newlines();
                elems.push(self.parse_expr(0)?);
                self.skip_newlines();
            }
        }
        Ok(elems)
    }
}
