Implement the real AST-walking evaluator in mlpl-eval, replacing the PoC.

Replace the PoC evaluate() with a proper evaluator that:

1. Environment type: HashMap<String, DenseArray> for variable bindings
2. eval_expr(expr, env) -> Result<DenseArray, EvalError>
   - IntLit -> DenseArray::from_scalar(n as f64)
   - FloatLit -> DenseArray::from_scalar(f)
   - Ident -> look up in env, error if undefined
   - ArrayLit -> evaluate each element, collect into DenseArray
     - Flat: [1, 2, 3] -> vector
     - Nested: [[1,2],[3,4]] -> matrix (flatten + reshape)
   - BinOp -> evaluate lhs and rhs, then apply operation (next step)
   - FnCall -> dispatch to runtime (step after next)
   - Assign -> evaluate value, store in env, return the value

3. eval_program(exprs, env) -> evaluate each statement, return last result

4. Update EvalError with proper variants:
   - UndefinedVariable(String)
   - EmptyInput
   - Unsupported(String) (keep for unimplemented features)
   - ArrayError(mlpl_array::ArrayError)

Keep BinOp and FnCall as Unsupported for now -- they get implemented in the next two steps.

TDD:
- Scalar literal: eval "42" -> scalar 42.0
- Array literal: eval "[1, 2, 3]" -> vector [1, 2, 3]
- Nested array: eval "[[1,2],[3,4]]" -> 2x2 matrix
- Variable: eval "x = 5" then "x" -> 5.0
- Undefined variable: error
- Multi-statement: "x = [1, 2, 3]\nx" -> vector

Requires: mlpl-parser parse() function from steps 001-004.

Allowed: crates/mlpl-eval/
May read: crates/mlpl-parser/, crates/mlpl-array/
