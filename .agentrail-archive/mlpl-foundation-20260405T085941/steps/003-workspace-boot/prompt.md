Bootstrap the Cargo workspace so that all crates compile.

1. Create root Cargo.toml as a workspace with members for all crates/ and apps/ directories.
2. For each crate (mlpl-core, mlpl-array, mlpl-parser, mlpl-runtime, mlpl-eval, mlpl-trace, mlpl-viz, mlpl-wasm, mlpl-ml, mlpl-cli), create:
   - Cargo.toml with correct name, version 0.1.0, edition 2024
   - Minimal src/lib.rs (or src/main.rs for mlpl-cli) with a module doc comment
3. For each app (mlpl-repl, mlpl-web, mlpl-lab), create:
   - Cargo.toml with correct name, version 0.1.0, edition 2024
   - Minimal src/main.rs with a placeholder main function
4. Wire up internal dependencies per the dependency flow:
   - mlpl-core: no internal deps
   - mlpl-array: depends on mlpl-core
   - mlpl-parser: depends on mlpl-core
   - mlpl-runtime: depends on mlpl-core, mlpl-array
   - mlpl-eval: depends on mlpl-core, mlpl-parser, mlpl-runtime, mlpl-array
   - mlpl-trace: depends on mlpl-core
   - mlpl-viz: depends on mlpl-core, mlpl-trace
   - mlpl-wasm: depends on mlpl-eval, mlpl-trace, mlpl-viz
   - mlpl-ml: depends on mlpl-array, mlpl-runtime
   - mlpl-cli: depends on mlpl-parser, mlpl-eval, mlpl-trace
   - apps depend on relevant crates as needed (minimal for now)
5. Verify: cargo check succeeds for the entire workspace
6. Verify: cargo test succeeds (no tests yet, but no compile errors)
7. Verify: cargo clippy --all-targets --all-features -- -D warnings passes
8. Add a root .gitignore for target/, Cargo.lock (or keep Cargo.lock -- your call for a workspace)

Do NOT add any real implementation code. Just stubs that compile.