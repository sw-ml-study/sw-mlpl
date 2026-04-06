Add arithmetic expression parsing with operator precedence to the parser.

Extend crates/mlpl-parser/src/parser.rs to handle:
1. Binary operators: +, -, *, /
2. Precedence: *, / bind tighter than +, -
3. Left-associative: 1 + 2 + 3 -> BinOp(BinOp(1,+,2),+,3)
4. Parenthesized grouping: (1 + 2) * 3

Use Pratt parsing or recursive descent with precedence climbing.

TDD:
- "1 + 2" -> BinOp(1, Add, 2)
- "1 + 2 * 3" -> BinOp(1, Add, BinOp(2, Mul, 3))
- "1 * 2 + 3" -> BinOp(BinOp(1, Mul, 2), Add, 3)
- "(1 + 2) * 3" -> BinOp(BinOp(1, Add, 2), Mul, 3)
- "1 - 2 - 3" -> BinOp(BinOp(1, Sub, 2), Sub, 3) (left-assoc)
- "[1, 2] + [3, 4]" -> BinOp(ArrayLit, Add, ArrayLit)

Allowed: crates/mlpl-parser/
