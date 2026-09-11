//! Top-level DECLARATION parsing that hangs off `Parser`: the contextual
//! `include "path"` statement and stacked `@word` annotations on a `def`.
//! Split out of `record_parser` to keep each module under the sw-checklist
//! function-count budget; the record-literal / field-access parsers stay
//! there.

use mlpl_core::Span;

use crate::parser::Parser;
use mlpl_lexer::{ParseError, TokenKind};
use mlpl_parser_ast::Expr;

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
