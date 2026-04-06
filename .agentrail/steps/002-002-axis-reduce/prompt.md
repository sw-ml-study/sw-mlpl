Implement axis-specific reduction in mlpl-runtime.

Currently reduce_add and reduce_mul reduce all elements to a scalar.
Add reduce_add_axis and reduce_mul_axis that reduce along a specific axis.

1. In mlpl-runtime, add:
   - reduce_add_axis(array, axis) -> reduce along the given axis
   - reduce_mul_axis(array, axis) -> reduce along the given axis

2. In mlpl-array, add DenseArray::reduce_axis(axis, op) that:
   - Validates axis < rank
   - Produces a new array with that axis removed
   - For a [2,3] matrix reduced along axis 0: result is [3] (column sums)
   - For a [2,3] matrix reduced along axis 1: result is [2] (row sums)

3. Wire as builtins: reduce_add(array, axis) with optional 2nd arg
   - 1 arg: reduce all (existing behavior)
   - 2 args: reduce along axis

TDD:
- reduce_add([[1,2,3],[4,5,6]], 0) -> [5, 7, 9] (column sums)
- reduce_add([[1,2,3],[4,5,6]], 1) -> [6, 15] (row sums)
- reduce_mul([[1,2],[3,4]], 0) -> [3, 8]
- reduce_add([1,2,3]) -> 6 (1 arg, unchanged behavior)
- Invalid axis -> error

Allowed: crates/mlpl-array/, crates/mlpl-runtime/, crates/mlpl-eval/
