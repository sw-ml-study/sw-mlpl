Implement dense array storage in mlpl-array. This builds on the Shape type from the previous step.

Implement in crates/mlpl-array/src/array.rs (or similar):
1. DenseArray struct -- holds Shape + Vec<f64> data
2. DenseArray::new(shape, data) -- constructor, validates data.len() == shape.element_count()
3. DenseArray::zeros(shape) -- create zero-filled array
4. DenseArray::from_scalar(value) -- rank-0 array
5. DenseArray::from_vec(data) -- rank-1 array
6. DenseArray::shape() -- returns &Shape
7. DenseArray::rank() -- delegates to shape
8. DenseArray::data() -- returns &[f64]
9. DenseArray::get(indices) -- multi-dimensional indexing with bounds check
10. DenseArray::set(indices, value) -- multi-dimensional indexed write with bounds check
11. Display impl -- format scalars, vectors (space-separated), matrices (row per line)

Write TDD-style:
- Test scalar creation and access
- Test vector creation, get, set
- Test matrix creation, row-major ordering, get/set
- Test bounds checking (out-of-bounds returns error)
- Test data length validation on construction
- Test Display output for scalar, vector, matrix

Verify:
- cargo test -p mlpl-array passes
- cargo clippy -p mlpl-array -- -D warnings passes

Allowed directories: crates/mlpl-array/, contracts/array-contract/