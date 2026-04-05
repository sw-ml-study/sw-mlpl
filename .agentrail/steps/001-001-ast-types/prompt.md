Define AST node types in mlpl-parser. These are parser-owned syntax nodes -- they do NOT depend on mlpl-array.

Create crates/mlpl-parser/src/ast.rs with:

1. Expr enum -- the core AST node:
   - IntLit(i64, Span)
   - FloatLit(f64, Span)
   - Ident(String, Span)
   - ArrayLit(Vec<Expr>, Span) -- [1, 2, 3] or [[1,2],[3,4]]
   - BinOp { op: BinOpKind, lhs: Box<Expr>, rhs: Box<Expr>, span: Span }
   - FnCall { name: String, args: Vec<Expr>, span: Span }
   - Assign { name: String, value: Box<Expr>, span: Span }

2. BinOpKind enum: Add, Sub, Mul, Div

3. Expr::span() method returning the Span for any variant

Add Expr to pub exports in lib.rs. Write tests for Expr construction and span extraction. Keep this step to type definitions only -- no parsing logic yet.

Allowed: crates/mlpl-parser/
