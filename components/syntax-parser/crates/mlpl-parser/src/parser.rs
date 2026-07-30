//! Parser: transforms token stream into AST.

use mlpl_core::Span;

use mlpl_lexer::{ParseError, Token, TokenKind, describe_kind};
use mlpl_parser_ast::{BinOpKind, Expr, TensorCtorKind};

/// Parse a token stream into a list of expression statements.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let mut p = Parser { tokens, pos: 0 };
    let mut stmts = Vec::new();
    p.skip_sep();
    while p.pos < p.tokens.len() && p.tokens[p.pos].kind != TokenKind::Eof {
        stmts.push(p.parse_statement()?);
        p.skip_sep();
    }
    Ok(stmts)
}

pub(crate) struct Parser<'a> {
    pub(crate) tokens: &'a [Token],
    pub(crate) pos: usize,
}

/// True if `kind` could start a fresh expression. Used by
/// `break` to decide whether a value follows or the break is
/// bare. The list is conservative: everything that can begin
/// an atom (literal, identifier, prefix op, opening delimiter,
/// keyword-headed expr like `if`) returns true; statement
/// terminators (`;`, newline, `}`, `)`, EOF, comma) and trailing
/// keywords (`else`) return false.
pub(crate) fn can_start_expr(kind: Option<&TokenKind>) -> bool {
    !matches!(
        kind,
        None | Some(
            TokenKind::Semicolon
                | TokenKind::Newline
                | TokenKind::RBrace
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::Comma
                | TokenKind::Else
                | TokenKind::Eof,
        )
    )
}

impl<'a> Parser<'a> {
    /// Parse a single statement (assignment, repeat, or expression).
    pub(crate) fn parse_statement(&mut self) -> Result<Expr, ParseError> {
        if self.tokens[self.pos].kind == TokenKind::Def {
            return self.parse_def();
        }
        if self.tokens[self.pos].kind == TokenKind::Return {
            return self.parse_return();
        }
        if self.tokens[self.pos].kind == TokenKind::Repeat {
            return self.parse_repeat(false);
        }
        if self.tokens[self.pos].kind == TokenKind::Train {
            return self.parse_repeat(true);
        }
        if self.tokens[self.pos].kind == TokenKind::For {
            return self.parse_for();
        }
        if self.tokens[self.pos].kind == TokenKind::Experiment {
            return self.parse_experiment();
        }
        if self.tokens[self.pos].kind == TokenKind::Device {
            return self.parse_device();
        }
        if matches!(self.tokens[self.pos].kind, TokenKind::Ident(_))
            && self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| t.kind == TokenKind::Equals)
        {
            let name_tok = &self.tokens[self.pos];
            let name = match &name_tok.kind {
                TokenKind::Ident(n) => n.clone(),
                _ => unreachable!(),
            };
            let start = name_tok.span;
            self.pos += 2;
            let value = self.parse_expr(0)?;
            let span = Span::new(start.start, value.span().end);
            return Ok(Expr::Assign {
                name,
                value: Box::new(value),
                span,
            });
        }
        if matches!(self.tokens[self.pos].kind, TokenKind::Ident(_))
            && self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| t.kind == TokenKind::Colon)
        {
            return self.parse_annotated_assign();
        }
        self.parse_expr(0)
    }

    /// Consume a token of the given kind or return `UnexpectedToken`.
    /// Returns the consumed token's span on success.
    pub(crate) fn expect(&mut self, kind: &TokenKind) -> Result<Span, ParseError> {
        if std::mem::discriminant(&self.tokens[self.pos].kind) == std::mem::discriminant(kind) {
            let span = self.tokens[self.pos].span;
            self.pos += 1;
            Ok(span)
        } else {
            Err(ParseError::UnexpectedToken {
                found: describe_kind(&self.tokens[self.pos].kind),
                span: self.tokens[self.pos].span,
            })
        }
    }

    /// Parse an expression with precedence climbing (min_prec=0 for full expr).
    pub(crate) fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let atom = self.parse_atom()?;
        let mut lhs = self.parse_postfix_chain(atom)?;
        loop {
            let Some((op, prec)) = self.tokens.get(self.pos).and_then(|t| match t.kind {
                TokenKind::Plus => Some((BinOpKind::Add, 1u8)),
                TokenKind::Minus => Some((BinOpKind::Sub, 1)),
                TokenKind::Star => Some((BinOpKind::Mul, 2)),
                TokenKind::Slash => Some((BinOpKind::Div, 2)),
                _ => None,
            }) else {
                break;
            };
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

    pub(crate) fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let tok = &self.tokens[self.pos];
        match &tok.kind {
            TokenKind::Minus => {
                let start = tok.span;
                self.pos += 1;
                let operand = self.parse_atom()?;
                let span = Span::new(start.start, operand.span().end);
                Ok(Expr::UnaryNeg {
                    operand: Box::new(operand),
                    span,
                })
            }
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
            TokenKind::StrLit(s) => {
                let e = Expr::StrLit(s.clone(), tok.span);
                self.pos += 1;
                Ok(e)
            }
            TokenKind::BuiltinRef(name) => {
                let name = name.clone();
                let span = tok.span;
                self.pos += 1;
                // `:name(args)` is a CALL of the referenced builtin --
                // the quoted form applied directly, so `:disp(g)` and
                // `disp(g)` are the same call. Bare `:name` stays the
                // first-class reference value.
                if self.is(TokenKind::LParen) {
                    return self.parse_fncall_after_name(name, span);
                }
                Ok(Expr::BuiltinRef(name, span))
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                let start = tok.span;
                self.pos += 1;
                // Tensor constructor: `param[...]` or `tensor[...]`
                if self.is(TokenKind::LBracket) {
                    let ctor_kind = match name.as_str() {
                        "param" => Some(TensorCtorKind::Param),
                        "tensor" => Some(TensorCtorKind::Tensor),
                        _ => None,
                    };
                    if let Some(kind) = ctor_kind {
                        return self.parse_tensor_ctor(kind, start);
                    }
                }
                // Namespaced call: ident ':' name '('
                if let Some(TokenKind::BuiltinRef(local)) =
                    self.tokens.get(self.pos).map(|t| &t.kind)
                {
                    let local = local.clone();
                    let full_name = format!("{name}:{local}");
                    self.pos += 1;
                    if self.is(TokenKind::LParen) {
                        self.pos += 1;
                        let mut args = Vec::new();
                        if !self.is(TokenKind::RParen) {
                            args.push(self.parse_expr(0)?);
                            while self.is(TokenKind::Comma) {
                                self.pos += 1;
                                args.push(self.parse_expr(0)?);
                            }
                        }
                        if !self.is(TokenKind::RParen) {
                            return Err(ParseError::UnclosedDelimiter {
                                open: "(".into(),
                                span: start,
                            });
                        }
                        let end = self.tokens[self.pos].span;
                        self.pos += 1;
                        return Ok(Expr::FnCall {
                            name: full_name,
                            args,
                            span: Span::new(start.start, end.end),
                        });
                    }
                    return Ok(Expr::Ident(full_name, start));
                }
                // Function call: ident '('
                if self.is(TokenKind::LParen) {
                    self.pos += 1; // skip '('
                    let mut args = Vec::new();
                    if !self.is(TokenKind::RParen) {
                        args.push(self.parse_expr(0)?);
                        while self.is(TokenKind::Comma) {
                            self.pos += 1;
                            args.push(self.parse_expr(0)?);
                        }
                    }
                    if !self.is(TokenKind::RParen) {
                        return Err(ParseError::UnclosedDelimiter {
                            open: "(".into(),
                            span: start,
                        });
                    }
                    let end = self.tokens[self.pos].span;
                    self.pos += 1;
                    Ok(Expr::FnCall {
                        name,
                        args,
                        span: Span::new(start.start, end.end),
                    })
                } else {
                    Ok(Expr::Ident(name, start))
                }
            }
            TokenKind::LBracket => self.parse_array_lit(),
            TokenKind::LBrace => {
                let open = tok.span;
                self.pos += 1;
                self.parse_record_lit_after_brace(open)
            }
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
            TokenKind::Device => self.parse_device(),
            TokenKind::If => self.parse_if(),
            TokenKind::Try => self.parse_try(),
            TokenKind::While => {
                let start = self.tokens[self.pos].span;
                self.pos += 1;
                let cond = self.parse_expr(0)?;
                let (body, end) = self.parse_braced_body()?;
                Ok(Expr::While {
                    cond: Box::new(cond),
                    body,
                    span: Span::new(start.start, end.end),
                })
            }
            TokenKind::Continue => {
                let span = self.tokens[self.pos].span;
                self.pos += 1;
                Ok(Expr::Continue { span })
            }
            TokenKind::Break => {
                let start = self.tokens[self.pos].span;
                self.pos += 1;
                let value = if can_start_expr(self.tokens.get(self.pos).map(|t| &t.kind)) {
                    let v = self.parse_expr(0)?;
                    let end = v.span().end;
                    Some((Box::new(v), end))
                } else {
                    None
                };
                let end = value.as_ref().map_or(start.end, |(_, e)| *e);
                Ok(Expr::Break {
                    value: value.map(|(v, _)| v),
                    span: Span::new(start.start, end),
                })
            }
            _ => Err(ParseError::UnexpectedToken {
                found: describe_kind(&tok.kind),
                span: tok.span,
            }),
        }
    }

    fn parse_tensor_ctor(&mut self, kind: TensorCtorKind, start: Span) -> Result<Expr, ParseError> {
        let open_span = self.tokens[self.pos].span;
        self.pos += 1; // skip '['
        let mut shape = Vec::new();
        if !self.is(TokenKind::RBracket) {
            shape.push(self.parse_expr(0)?);
            while self.is(TokenKind::Comma) {
                self.pos += 1;
                shape.push(self.parse_expr(0)?);
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
        Ok(Expr::TensorCtor {
            kind,
            shape,
            span: Span::new(start.start, close_span.end),
        })
    }

    /// Parse `(args...)` after a known callee name (identifier or
    /// builtin reference) into an `Expr::FnCall`. The opening paren
    /// is NOT yet consumed.
    fn parse_fncall_after_name(&mut self, name: String, start: Span) -> Result<Expr, ParseError> {
        self.pos += 1; // skip '('
        let mut args = Vec::new();
        if !self.is(TokenKind::RParen) {
            args.push(self.parse_expr(0)?);
            while self.is(TokenKind::Comma) {
                self.pos += 1;
                args.push(self.parse_expr(0)?);
            }
        }
        if !self.is(TokenKind::RParen) {
            return Err(ParseError::UnclosedDelimiter {
                open: "(".into(),
                span: start,
            });
        }
        let end = self.tokens[self.pos].span;
        self.pos += 1;
        Ok(Expr::FnCall {
            name,
            args,
            span: Span::new(start.start, end.end),
        })
    }

    fn parse_array_lit(&mut self) -> Result<Expr, ParseError> {
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
    fn parse_array_elems(&mut self) -> Result<Vec<Expr>, ParseError> {
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

    /// Parse `if cond { then } else { else }`. The `else`
    /// clause is required (no dangling-if). Saga 31 step 004.
    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1; // skip 'if'
        let cond = self.parse_expr(0)?;
        let (then_body, then_end) = self.parse_braced_body()?;
        // `else` is optional: a bare `if cond { body }` is a valid
        // statement-position conditional whose value is the body when
        // taken and `0` otherwise (issue #6 / C3).
        let (else_body, end) = if self.is(TokenKind::Else) {
            self.pos += 1;
            let (eb, ee) = self.parse_braced_body()?;
            (eb, ee.end)
        } else {
            (Vec::new(), then_end.end)
        };
        Ok(Expr::If {
            cond: Box::new(cond),
            then_body,
            else_body,
            span: Span::new(start.start, end),
        })
    }

    fn parse_repeat(&mut self, is_train: bool) -> Result<Expr, ParseError> {
        let start = self.tokens[self.pos].span;
        self.pos += 1; // skip 'repeat' or 'train'
        let count = self.parse_expr(0)?;
        let (body, end) = self.parse_braced_body()?;
        let span = Span::new(start.start, end.end);
        let count = Box::new(count);
        Ok(if is_train {
            Expr::Train { count, body, span }
        } else {
            Expr::Repeat { count, body, span }
        })
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

    /// Skip newline tokens only (not `;`). Used inside bracketed
    /// constructs (array literals) where newlines are insignificant --
    /// commas separate elements -- so a matrix can span lines (C2).
    pub(crate) fn skip_newlines(&mut self) {
        while self.pos < self.tokens.len() && self.tokens[self.pos].kind == TokenKind::Newline {
            self.pos += 1;
        }
    }
}
