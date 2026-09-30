//! The bracket- and keyword-introduced atoms: array literals
//! (`[e, e, ...]`), prefix `not`, and `try { } catch e { }`. Lives in a
//! sibling module to keep `parser.rs` under the sw-checklist file-LOC
//! budget (same pattern as `record_parser.rs` / `stmts.rs`). The entry
//! points hang off `Parser` and are called from `parse_atom` when it sees
//! `[`, `not`, or `try`.

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

    /// Prefix `not <operand>` (cursor on `not`). Desugars to `eq(x, 0)`: a
    /// 0/1 result that works elementwise, as a stop-gradient mask inside
    /// `grad`, and in compiled programs, with no new AST form. The operand
    /// binds at the comparison level, so `not a < b` is `not (a < b)` and
    /// `not a and b` is `(not a) and b` -- Python's precedence.
    pub(crate) fn parse_not(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1;
        let operand = self.parse_expr(crate::parser::COMPARISON_PREC)?;
        let span = Span::new(start.start, operand.span().end);
        Ok(Expr::FnCall {
            name: "eq".into(),
            args: vec![operand, Expr::IntLit(0, span)],
            span,
        })
    }

    /// Parse `try { body } catch <ident> { handler }`. Both
    /// braces are required; the binding is a plain ident.
    pub(crate) fn parse_try(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1; // skip 'try'
        let (body, _) = self.parse_braced_body()?;
        self.expect(&TokenKind::Catch)?;
        let tok = &self.tokens[self.pos];
        let TokenKind::Ident(binding) = &tok.kind else {
            return Err(ParseError::UnexpectedToken {
                found: mlpl_lexer::describe_kind(&tok.kind),
                span: tok.span,
            });
        };
        let binding = binding.clone();
        self.pos += 1;
        let (handler, end) = self.parse_braced_body()?;
        Ok(Expr::TryCatch {
            body,
            binding,
            handler,
            span: Span::new(start.start, end.end),
        })
    }
}
