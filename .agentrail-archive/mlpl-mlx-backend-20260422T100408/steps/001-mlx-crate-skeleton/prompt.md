Phase 1 step 001: mlpl-mlx crate skeleton + first primitive.

Stand up the MLX runtime target as a sibling to `mlpl-rt`.

1. New crate `crates/mlpl-mlx` with a `mlx` Cargo feature that
   pulls in `mlx-rs` (pin a known-good version in
   `Cargo.toml`). Add the crate to the workspace.
2. Port exactly one primitive: `matmul(a, b)`. Signature mirrors
   `mlpl-rt::matmul` so the two runtimes are drop-in swappable.
   Internally wrap MLX's `matmul` and hand back a
   `DenseArray` + `LabeledShape` (from `mlpl-core`) matching the
   CPU path's output shape and labels.
3. Parity test: `[8, 4] @ [4, 8]` fixture runs through both
   `mlpl-rt::matmul` and `mlpl-mlx::matmul` and the outputs
   agree bit-for-bit, or within a documented fp32 tolerance
   (decide during TDD and write the tolerance into the test).
4. Gate all MLX-backed tests behind `#[cfg(all(target_os =
   "macos", target_arch = "aarch64", feature = "mlx"))]` (or the
   closest equivalent `mlx-rs` supports). On non-Apple hosts
   `cargo test -p mlpl-mlx` still compiles and passes.
5. TDD: parity test first (red -> green -> refactor).
6. Quality gates + `/mw-cp`. Commit message references step 001.
