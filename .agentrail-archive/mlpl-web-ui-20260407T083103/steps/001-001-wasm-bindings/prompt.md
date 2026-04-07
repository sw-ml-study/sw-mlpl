Implement WASM bindings in mlpl-wasm crate.

1. Add wasm-bindgen dependency to mlpl-wasm/Cargo.toml
2. Set crate-type = ["cdylib", "rlib"] in Cargo.toml
3. Implement a public wasm-bindgen API in src/lib.rs:
   - eval_line(input: &str) -> String
     Parses and evaluates a single MLPL expression, returns the
     result as a display string or an error message prefixed with "error: "
   - create_session() -> WasmSession (or similar)
     Creates a persistent environment so variables survive across calls
   - session_eval(session, input) -> String
     Evaluates within a session's environment
4. Write tests that verify eval_line works for:
   - Scalar arithmetic: "1 + 2" -> "3"
   - Array ops: "[1,2,3] * 10" -> "10 20 30"
   - Variable assignment: "x = 42" then "x + 1" -> "43"
   - Built-in functions: "iota(5)" -> "0 1 2 3 4"
   - Error cases: "foo(1)" -> "error: ..."
5. Ensure cargo test passes for the wasm crate
6. Ensure clippy and fmt pass

Note: wasm-pack test or wasm-bindgen-test may be needed for
full WASM testing. At minimum, test the eval logic natively.

Allowed: crates/mlpl-wasm/