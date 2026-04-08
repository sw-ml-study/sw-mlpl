Add seeded random number generation built-ins.

1. random(seed, shape) -> array of uniform [0, 1) values with the given shape (shape is a length-K vector of dims).
2. randn(seed, shape) -> array of standard-normal values, same shape semantics. Use Box-Muller from the uniform stream.
3. seed is a scalar integer; same seed must produce identical output (deterministic).
4. Implement a simple xorshift64 PRNG as a tiny private module in mlpl-runtime; do NOT add an external rand crate dependency.
5. TDD: write failing tests for (a) shape correctness, (b) determinism (same seed -> same output), (c) value range for uniform, (d) approximate mean/variance for randn over a large sample.
6. Update docs/lang-reference.md and docs/usage.md with the new built-ins.
7. Allowed: crates/mlpl-runtime, crates/mlpl-eval (only if needed), docs/