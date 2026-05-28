# Warning/FAIL Ratchet-Down Spike

Focused tech-debt reduction targeting FAILs from this
session's work. Priority: split paths.rs (1162 lines),
shrink over-LOC functions in runtime and web crates.

## Steps

1. Split paths.rs into per-path files. Reduce File LOC FAIL.
2. Shrink over-50-LOC functions (math_builtins, ast_fmt, stmts).
3. language-status + saga close.