Implement the Shape type in mlpl-array. This is task ARRAY-SHAPE-001 from the task packets.

Implement in crates/mlpl-array/src/shape.rs:
1. Shape struct -- wraps a Vec<usize> of dimensions
2. Shape::new(dims) -- constructor with validation
3. Shape::rank() -- number of dimensions
4. Shape::dims() -- slice of dimensions
5. Shape::element_count() -- product of dimensions (1 for scalar/empty)
6. Shape::scalar() -- creates rank-0 shape
7. Shape::vector(len) -- creates rank-1 shape
8. Shape::matrix(rows, cols) -- creates rank-2 shape
9. Display impl -- e.g., "[2, 3]" or "[]" for scalar

Error type in crates/mlpl-array/src/error.rs:
1. ArrayError enum with at least ShapeError variant
2. Descriptive error messages

Write TDD-style following contracts/array-contract/README.md:
- Test shape construction (scalar, vector, matrix, n-D)
- Test rank computation
- Test element count computation
- Test Display output
- Test error cases if any validation exists

Verify:
- cargo test -p mlpl-array passes
- cargo clippy -p mlpl-array -- -D warnings passes

Allowed directories: crates/mlpl-array/, contracts/array-contract/
Do NOT modify: crates/mlpl-parser/, crates/mlpl-eval/, apps/, root Cargo.toml