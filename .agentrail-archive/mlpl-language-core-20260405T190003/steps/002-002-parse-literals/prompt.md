Implement parser for literals and array literals.

Create crates/mlpl-parser/src/parser.rs with a Parser struct that takes a token slice and produces Expr nodes.

Parse these forms:
1. Integer literal: 42 -> Expr::IntLit(42, span)
2. Float literal: 1.5 -> Expr::FloatLit(1.5, span)
3. Negative literals: -3 -> Expr::IntLit(-3, span)
4. Array literal: [1, 2, 3] -> Expr::ArrayLit(vec![...], span)
5. Nested array: [[1, 2], [3, 4]] -> Expr::ArrayLit(vec![ArrayLit(...), ...], span)
6. Identifier: x -> Expr::Ident("x", span)
7. Parenthesized expression: (expr) -> inner expr

Add a pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> that parses a full token stream into a list of statements (expressions separated by newlines/semicolons).

TDD: test each literal form, array nesting, identifier, parens, multi-statement parsing, error on unexpected token.

Allowed: crates/mlpl-parser/
