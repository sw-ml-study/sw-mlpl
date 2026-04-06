Implement element-wise arithmetic and scalar broadcasting in the evaluator.

Add to mlpl-eval (or mlpl-array if the operations belong there):

1. Element-wise binary operations on DenseArray:
   - add, sub, mul, div for same-shape arrays
   - Each operates element-by-element, producing a new array

2. Scalar broadcasting:
   - scalar op array -> apply scalar to every element
   - array op scalar -> apply scalar to every element
   - Result has the array's shape

3. Wire BinOp evaluation in eval_expr:
   - Evaluate lhs and rhs
   - If shapes match: element-wise
   - If one is scalar: broadcast
   - Otherwise: ShapeMismatch error

4. Division by zero: produce f64::INFINITY or NaN (IEEE semantics, no error)

TDD:
- "1 + 2" -> scalar 3.0
- "[1, 2, 3] + [4, 5, 6]" -> vector [5, 7, 9]
- "[1, 2, 3] * 10" -> vector [10, 20, 30]
- "10 * [1, 2, 3]" -> vector [10, 20, 30]
- "[1, 2] + [1, 2, 3]" -> ShapeMismatch error
- "1 + 2 * 3" -> scalar 7.0 (precedence)
- "x = [1, 2, 3]\nx + 1" -> vector [2, 3, 4]

Allowed: crates/mlpl-eval/, crates/mlpl-array/
