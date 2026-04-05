Implement reshape and transpose operations in mlpl-array.

Reshape (crates/mlpl-array/src/reshape.rs or in array.rs):
1. DenseArray::reshape(new_shape) -> Result<DenseArray, ArrayError>
   - Preserves element order (row-major)
   - Succeeds only when element counts match
   - Returns new array with same data, different shape
   - Error on incompatible shapes

Transpose (crates/mlpl-array/src/transpose.rs or in array.rs):
1. DenseArray::transpose() -> DenseArray
   - Reverses axis order
   - For matrix: swaps rows and columns
   - For vector: no-op (or returns same vector)
   - For scalar: no-op
   - Must reorder data to maintain row-major layout

Write TDD-style:
- Test reshape: vector to matrix, matrix to vector, matrix to different matrix
- Test reshape error: incompatible element counts
- Test transpose: matrix transpose correctness
- Test transpose: vector and scalar identity behavior
- Test round-trip: reshape then transpose, verify data integrity

Verify:
- cargo test -p mlpl-array passes
- cargo clippy -p mlpl-array -- -D warnings passes

Allowed directories: crates/mlpl-array/, contracts/array-contract/