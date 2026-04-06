Implement matrix multiplication in mlpl-array and wire as a built-in.

1. In mlpl-array, add DenseArray::matmul(&self, other: &DenseArray):
   - self is [m, k], other is [k, n] -> result is [m, n]
   - Inner dimensions must match
   - Uses standard row-by-column dot product
   - Also handle: matrix * vector ([m, k] * [k]) -> [m] (mat-vec product)

2. In mlpl-runtime, add "matmul" built-in:
   - matmul(A, B) -> matrix product

TDD:
- matmul([[1,2],[3,4]], [[5,6],[7,8]]) -> [[19,22],[43,50]]
- matmul identity: matmul([[1,0],[0,1]], [[5,6],[7,8]]) -> [[5,6],[7,8]]
- matmul mat-vec: matmul([[1,2],[3,4]], [5,6]) -> [17, 39]
- matmul dimension mismatch -> error
- matmul with iota+reshape for larger test

Allowed: crates/mlpl-array/, crates/mlpl-runtime/, crates/mlpl-eval/
