Add unary negation support to the parser and evaluator.

Currently "-3" lexes as IntLit(-3) which works for literals, but
"-(x + 1)" or "-x" don't parse correctly because the lexer treats
minus as part of a number or as a binary operator.

1. Add Expr::UnaryNeg { operand: Box<Expr>, span: Span } to the AST

2. In the parser, handle unary minus in parse_atom:
   - If current token is Minus and it's NOT preceded by a value,
     consume it and parse the next atom, wrapping in UnaryNeg
   - "-x" -> UnaryNeg(Ident("x"))
   - "-(1 + 2)" -> UnaryNeg(BinOp(1, Add, 2))
   - "-[1, 2, 3]" -> UnaryNeg(ArrayLit)

3. In the evaluator, handle UnaryNeg:
   - Evaluate operand, then negate every element (multiply by -1)

TDD:
- Parse "-x" -> UnaryNeg(Ident)
- Parse "-(1 + 2)" -> UnaryNeg(BinOp)
- Eval "-5" -> scalar -5.0
- Eval "x = 3; -x" -> scalar -3.0
- Eval "-[1, 2, 3]" -> vector [-1, -2, -3]
- Eval "1 + -2" -> scalar -1.0

Allowed: crates/mlpl-parser/, crates/mlpl-eval/
