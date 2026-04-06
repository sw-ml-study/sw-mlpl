Implement dot product for vectors in mlpl-array and wire as a built-in.

1. In mlpl-array, add DenseArray::dot(&self, other: &DenseArray):
   - Both must be rank-1 (vectors) with same length
   - Returns scalar (sum of element-wise products)
   - Error on rank mismatch or length mismatch

2. In mlpl-runtime, add "dot" built-in:
   - dot(a, b) -> scalar dot product

TDD:
- dot([1,2,3], [4,5,6]) -> 32 (1*4 + 2*5 + 3*6)
- dot([1,0], [0,1]) -> 0 (orthogonal)
- dot([2], [3]) -> 6 (length-1)
- dot([1,2], [1,2,3]) -> error (length mismatch)
- dot([[1,2],[3,4]], [1,2]) -> error (rank mismatch)

Allowed: crates/mlpl-array/, crates/mlpl-runtime/, crates/mlpl-eval/
