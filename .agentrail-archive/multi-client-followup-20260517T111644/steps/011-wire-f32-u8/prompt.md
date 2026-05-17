Saga 21.5 step 011: wire-f32-u8.

Goal: services/mlpl-mlx-serve/src/wire.rs grows two dtype slots beyond the current f64-only path. Add DTYPE_F32 = 1, DTYPE_U8 = 2. The peer protocol uses the dtype the orchestrator sends; the orchestrator picks dtype based on the source MLPL array. Image tensors materialize as u8 at the source and stay u8 over the wire until the first arithmetic op upgrades them.

TDD (Red/Green/Refactor):

1. RED tests:
   - services/mlpl-mlx-serve/tests/wire_dtype_tests.rs (new): round-trip f32 fixture (encode + decode preserves precision within f32 epsilon); round-trip u8 fixture (preserves byte-exact); rejecting an unknown dtype tag yields a clear error.
   - crates/mlpl-eval/tests/wire_promotion_tests.rs (new, if scope allows): u8 + f64 promotes to f64 on the wire (within tolerance vs in-process); f32 + f32 stays f32.

2. GREEN:
   - services/mlpl-mlx-serve/src/wire.rs: extend the dtype enum from f64-only to a 3-variant tagged union. Encoder picks the right tag from the orchestrator's source array; decoder dispatches on the tag.
   - Orchestrator side (crates/mlpl-serve/src/peers.rs encode_bindings or wherever DenseArray is encoded): pick dtype based on the source array's effective dtype. DenseArray is f64 internally today; for now only image-loading paths produce u8-tagged arrays. Document the promotion ladder.
   - Wire protocol bump if needed (with a version byte at the start).

3. REFACTOR: keep sw-checklist budgets. mlpl-mlx-serve is a separate crate; should have headroom. Document the wire-format contract changes in contracts/mlx-serve-contract/wire.md (or wherever it lives).

Quality gates per /mw-cp: cargo test, cargo clippy, cargo fmt, markdown-checker, sw-checklist (held). Update wire contract.

Out of scope: docs + release (012/013); cross-device promotion semantics beyond the basic ladder; image-tensor loading (a future saga); ViT (saga 29).