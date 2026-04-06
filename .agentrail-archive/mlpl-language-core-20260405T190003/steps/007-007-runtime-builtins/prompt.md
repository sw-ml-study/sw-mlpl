Implement the built-in function registry in mlpl-runtime and wire it into the evaluator.

1. In crates/mlpl-runtime/src/lib.rs, create a dispatch function:
   call_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError>

2. Implement these built-in functions:
   - reshape(array, shape_array) -> reshape array to shape derived from shape_array
   - transpose(array) -> transpose the array
   - shape(array) -> return shape as a 1-D vector of dimensions
   - rank(array) -> return rank as a scalar
   - iota(n) -> vector [0, 1, 2, ..., n-1] where n is a scalar integer

3. RuntimeError enum:
   - UnknownFunction(String)
   - ArityMismatch { func, expected, got }
   - InvalidArgument { func, reason }
   - ArrayError(mlpl_array::ArrayError)

4. Wire FnCall evaluation in mlpl-eval:
   - Evaluate each argument
   - Call call_builtin(name, args)
   - Propagate errors

TDD:
- "iota(5)" -> [0, 1, 2, 3, 4]
- "shape([1, 2, 3])" -> [3]
- "rank([1, 2, 3])" -> scalar 1.0
- "reshape(iota(6), [2, 3])" -> 2x3 matrix
- "transpose(reshape(iota(6), [2, 3]))" -> 3x2 matrix
- "unknown(1)" -> UnknownFunction error
- "reshape(iota(6), [2, 2])" -> ShapeMismatch error

Allowed: crates/mlpl-runtime/, crates/mlpl-eval/
May read: crates/mlpl-array/, crates/mlpl-parser/
