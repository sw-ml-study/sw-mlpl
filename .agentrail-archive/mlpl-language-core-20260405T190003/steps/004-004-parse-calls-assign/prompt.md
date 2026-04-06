Add function call and assignment parsing to the parser.

Extend crates/mlpl-parser/src/parser.rs to handle:

1. Function calls: reshape(x, [2, 2]) -> Expr::FnCall { name, args, span }
   - An identifier followed by '(' triggers function call parsing
   - Comma-separated argument list
   - Nested calls: shape(reshape(x, [2, 2]))

2. Assignment: x = expr -> Expr::Assign { name, value, span }
   - An identifier followed by '=' triggers assignment parsing
   - Right-hand side is any expression (including function calls, arithmetic)
   - Assignment is right-associative and lowest precedence

3. Distinguish identifier from function call:
   - "x" alone is Expr::Ident
   - "x(" starts a function call
   - "x =" starts an assignment

TDD:
- "reshape(x, [2, 2])" -> FnCall with 2 args
- "shape(x)" -> FnCall with 1 arg
- "iota(6)" -> FnCall with 1 IntLit arg
- "x = 42" -> Assign { name: "x", value: IntLit(42) }
- "x = [1, 2, 3]" -> Assign with ArrayLit value
- "x = 1 + 2" -> Assign with BinOp value
- "x = reshape(y, [2, 3])" -> Assign with FnCall value
- Multi-statement: "x = 1\ny = x + 1" -> two Assign exprs

Allowed: crates/mlpl-parser/
