Scripting saga step 001: add print(value) and eprint(value) builtins.

print(value) writes the value's display form (the same format the REPL prints for the terminal) to stdout, followed by a newline. eprint(value) writes to stderr. Both return their argument UNCHANGED so they compose naturally into expressions: x = print(some_computation) both binds x and shows the value.

Why return-the-argument: MLPL is expression-only. A print that returned () or 0 would force callers to wrap it in a sequencing block or discard its value awkwardly. Returning the input preserves the expression model. (Precedent: Rust's dbg!() macro.)

TDD (Red/Green/Refactor):
- RED: write tests in crates/mlpl-runtime/tests/ that exercise print + eprint on a few sample values (scalar, vector, matrix) and assert (a) the output goes to the right stream and (b) the return value equals the input. Use a captured-writer test harness if the runtime supports it; if not, add a tiny one.
- GREEN: register 'print' and 'eprint' in crates/mlpl-runtime/src/builtins.rs and route them to small impls that take a writer + Value and emit the display form.
- REFACTOR: the writer abstraction may already exist (env.set_writer or similar); reuse it instead of building a new one. Keep the impl under the 50-LOC-per-fn budget.

Quality gates: cargo test -p mlpl-runtime; cargo clippy -p mlpl-runtime --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md with a new row for print(value) and eprint(value).

No web changes; no pages rebuild.

After this step ships the script writer has a way to surface intermediate values explicitly -- a foundational scripting primitive that the rest of the saga can build on.