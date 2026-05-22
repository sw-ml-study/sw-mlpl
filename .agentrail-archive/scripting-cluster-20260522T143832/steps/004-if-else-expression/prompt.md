Scripting saga step 004: add 'if cond { then } else { else }' as an expression.

Surface form: if cond { then_expr } else { else_expr }. The else clause is REQUIRED (no dangling-if). Returns the value of the chosen branch. cond is truthy iff non-zero (matches existing convention: any Value::Number != 0.0 is truthy; Value::Result is truthy when .ok is true). Non-truthy values include 0.0 and Err(_) Results.

The expression model: 'let x = if flag { 42 } else { 99 }' binds x to either 42 or 99 based on flag. No 'if' without 'else' for now -- a missing else would force a unit value, which MLPL does not have.

TDD:
- RED: parser tests in crates/mlpl-parser/tests/ that parse 'if 1 { 42 } else { 99 }' into the expected AST shape, and a test that 'if 1 { 42 }' (no else) is a parse error.
- RED: eval tests in crates/mlpl-eval/tests/ asserting if 1 { 42 } else { 99 } evals to 42; if 0 { 42 } else { 99 } evals to 99; nested if works; cond on a Result-typed value uses .ok for truthiness.
- GREEN: add Expr::If { cond, then, else_ } to crates/mlpl-parser/src/ast.rs; extend the parser to recognize the surface form; add the eval rule in crates/mlpl-eval/src/eval.rs.

Quality gates: cargo test workspace; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md with a new section on if/else.

Glossary: add an 'If expression' entry to docs/glossary.md. Run the [[term]] cross-link sweep script (/tmp/sweep_glossary.py from saga 29 step 028's work, or rewrite if missing) to wire references.

After this step ships scripts can branch on a scalar flag.