Scripting saga step 002: add to_number(s), to_int(s), env(name) builtins.

All three return Value::Result so the caller is forced to handle the failure modes explicitly.

- to_number(s): parse a string as f64. Ok(n) on success; Err('to_number: cannot parse "abc" as a number') on failure.
- to_int(s): same but rejects non-integer strings. Err('to_int: "3.5" is not an integer') on a floating-point literal; Err('to_int: cannot parse "xyz" as an integer') on a non-numeric.
- env(name): read OS env var. Ok('value') if set; Err('env: MODEL_PATH not set') if unset. Use std::env::var.

TDD:
- RED: tests in crates/mlpl-runtime/tests/ (or new file scripting_builtins_tests.rs) exercising the happy path + each specific error class. For env() set + unset a temporary env var via std::env::set_var inside the test.
- GREEN: register the three names in crates/mlpl-runtime/src/builtins.rs and route to small per-builtin impls. Each impl is one match arm + an explicit Err construction.
- REFACTOR: factor any shared 'wrap Result' helper if the three impls have parallel structure.

Quality gates: cargo test -p mlpl-runtime; cargo clippy -p mlpl-runtime --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md with three new rows.

After this step ships scripts can convert their CLI args to numbers and read configuration from the environment.