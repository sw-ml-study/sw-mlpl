//! Interpreter vs compiler parity tests (compile-to-rust step 006).
//!
//! See `tests/parity_tests.rs` for the harness and cases. This crate
//! exists purely to host that integration-test file so the slow
//! rustc-based parity run is opt-in via `MLPL_PARITY_TESTS=1` and
//! lives in its own build unit. Intentionally empty.
