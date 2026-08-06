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
//!   `parse_atom` returns. Repeatedly consumes `.ident` and
//!   wraps the running expression in `FieldAccess`. Binds
//!   tighter than every infix binop (`f(x).y + z` parses as
//!   `(f(x).y) + z`).
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
    let kw = |s: &str| Some(s.to_string());
    match kind {
        TokenKind::Ident(name) => Some(name.clone()),
        TokenKind::Repeat => kw("repeat"),
        TokenKind::Train => kw("train"),
        TokenKind::For => kw("for"),
        TokenKind::In => kw("in"),
        TokenKind::Experiment => kw("experiment"),
        TokenKind::Device => kw("device"),
        TokenKind::If => kw("if"),
        TokenKind::Else => kw("else"),
        TokenKind::While => kw("while"),
        TokenKind::Break => kw("break"),
        TokenKind::Continue => kw("continue"),
        TokenKind::Try => kw("try"),
        TokenKind::Catch => kw("catch"),
        TokenKind::Def => kw("def"),
        TokenKind::Return => kw("return"),
        _ => None,
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
    /// Parse `try { body } catch <ident> { handler }`. Both
    /// braces are required; the binding is a plain ident. Spike
    /// step 011.
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

impl Parser<'_> {
    /// `Ident("include")` immediately followed by a string
    /// literal. That token sequence is a parse error today, so
    /// claiming it costs nothing: `include` stays a legal
    /// variable name everywhere else (contextual, like keyword
    /// field names above).
    pub(crate) fn include_pattern(&self) -> bool {
        matches!(&self.tokens[self.pos].kind, TokenKind::Ident(n) if n == "include")
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::StrLit(_))
            )
    }

    /// Consume a top-level `include "path"` declaration; `None`
    /// when the upcoming tokens are not the include pattern.
    pub(crate) fn parse_include_top(&mut self) -> Option<Result<Expr, ParseError>> {
        if !self.include_pattern() {
            return None;
        }
        let start = self.tokens[self.pos].span;
        self.pos += 1;
        let TokenKind::StrLit(path) = &self.tokens[self.pos].kind else {
            unreachable!("include_pattern guarantees a string literal");
        };
        let (path, end) = (path.clone(), self.tokens[self.pos].span);
        self.pos += 1;
        Some(Ok(Expr::Include(path, Span::new(start.start, end.end))))
    }
}

impl Parser<'_> {
    /// Parse stacked `@word [record-literal | string-literal]`
    /// annotations and the `def u:` they attach to. `@` is a
    /// GENERAL annotation namespace: any word is legal; payloads
    /// are one record or string literal. Annotations attach ONLY
    /// to a following def.
    pub(crate) fn parse_annotated_def(&mut self) -> Result<Expr, ParseError> {
        let mut annotations: Vec<(String, Option<Expr>)> = Vec::new();
        while self.tokens[self.pos].kind == TokenKind::At {
            let at_span = self.tokens[self.pos].span;
            self.pos += 1;
            let TokenKind::Ident(word) = &self.tokens[self.pos].kind else {
                return Err(ParseError::UnexpectedToken {
                    found: "an annotation needs a word: `@test`, `@formula ...`".into(),
                    span: at_span,
                });
            };
            let word = word.clone();
            self.pos += 1;
            annotations.push((word, self.parse_annotation_payload()?));
            self.skip_sep();
        }
        if self.tokens[self.pos].kind != TokenKind::Def {
            return Err(ParseError::UnexpectedToken {
                found: "annotations attach to the NEXT `def u:...` definition".into(),
                span: self.tokens[self.pos].span,
            });
        }
        let def = self.parse_def()?;
        let Expr::FnDef {
            name,
            params,
            body,
            span,
            ..
        } = def
        else {
            unreachable!("parse_def returns FnDef");
        };
        Ok(Expr::FnDef {
            name,
            params,
            body,
            annotations,
            span,
        })
    }

    /// One optional annotation payload: a `{...}` record literal
    /// or a string literal on the same line.
    fn parse_annotation_payload(&mut self) -> Result<Option<Expr>, ParseError> {
        match &self.tokens[self.pos].kind {
            TokenKind::LBrace => {
                let brace = self.tokens[self.pos].span;
                self.pos += 1;
                Ok(Some(self.parse_record_lit_after_brace(brace)?))
            }
            TokenKind::StrLit(s) => {
                let e = Expr::StrLit(s.clone(), self.tokens[self.pos].span);
                self.pos += 1;
                Ok(Some(e))
            }
            _ => Ok(None),
        }
    }
}
