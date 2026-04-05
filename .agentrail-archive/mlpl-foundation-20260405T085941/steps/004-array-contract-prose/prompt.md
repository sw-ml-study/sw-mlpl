Write the detailed array-contract prose spec in contracts/array-contract/README.md.

This is the most important contract for the project. Cover:

1. Shape model:
   - Shape is an ordered list of non-negative dimensions
   - Rank = number of dimensions
   - Scalar has rank 0 (empty shape)
   - Vector has rank 1
   - Matrix has rank 2
   - Total element count = product of dimensions (empty shape = 1 element for scalar)

2. Dense storage:
   - Row-major contiguous storage
   - Element types: at minimum f64 for MVP
   - Strides derived from shape (not stored independently for now)

3. Indexing:
   - 0-origin
   - Bounds checking required
   - Multi-dimensional indexing

4. Reshape:
   - Preserves element order (row-major)
   - Succeeds only when total element count matches
   - Returns explicit error on incompatible target shape

5. Transpose:
   - Reverses axis order
   - For matrix: swaps rows and columns

6. Broadcasting/pervasion rules (high-level, detailed later):
   - Scalar extends to any shape
   - Matching shapes operate element-wise
   - Incompatible shapes produce an error

7. Error cases:
   - ShapeError: incompatible reshape, out-of-bounds index, etc.
   - Each error variant should have a clear description

8. What this contract does NOT cover:
   - Parser syntax for array literals
   - Evaluator semantics
   - Trace serialization
   - Boxed/nested arrays (post-MVP)