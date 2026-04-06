Add reduce_add and reduce_mul built-in functions to mlpl-runtime.

1. reduce_add(array) -> scalar that is the sum of all elements
2. reduce_mul(array) -> scalar that is the product of all elements

These reduce an array of any rank to a single scalar value.

TDD:
- "reduce_add([1, 2, 3])" -> scalar 6.0
- "reduce_mul([1, 2, 3, 4])" -> scalar 24.0
- "reduce_add(iota(5))" -> scalar 10.0 (0+1+2+3+4)
- "reduce_add(reshape(iota(6), [2, 3]))" -> scalar 15.0
- "reduce_add(42)" -> scalar 42.0 (scalar identity)
- "reduce_mul([2, 0, 3])" -> scalar 0.0

Allowed: crates/mlpl-runtime/
May read: crates/mlpl-eval/, crates/mlpl-array/
