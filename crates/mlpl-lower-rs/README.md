# mlpl-lower-rs

Lower MLPL AST to a Rust `proc_macro2::TokenStream`.

Shared codegen for the future `mlpl!` proc macro (step 005) and
`mlpl build` subcommand (step 008). Walks an `mlpl_parser::Expr`
AST and emits Rust code that, when compiled, produces the same
numeric result as the interpreter.

## Phase coverage

- **Phase 1 (step 002, this step):** scalar arithmetic only --
  `IntLit`, `FloatLit`, `UnaryNeg`, `BinOp(+, -, *, /)`. Emits
  native `f64` expressions wrapped at the outermost level in
  `mlpl_rt::DenseArray::from_scalar(...)`.
- **Phase 2 (steps 003-004):** array literals, function calls for
  the phase-1 `mlpl-rt` primitives, variable bindings, and
  label-aware lowering with static matmul contraction checks.
- **Phase 3 (step 005):** exposed through the `mlpl!` proc macro
  with span-preserved error reporting.

## Testing approach

Two test suites:

1. **`lower_scalar_tests.rs`** -- fast string-match assertions on
   the emitted `TokenStream`. These run on every `cargo test` and
   give feedback in milliseconds.
2. **`compile_tests.rs`** -- slow end-to-end sanity check. Lowers
   source to tokens, writes a tiny Rust crate to a temp dir that
   depends on `mlpl-rt` via a path dependency, shells out to
   `cargo build --release`, runs the binary, and parses the
   numeric stdout. Gated behind the `MLPL_LOWER_RS_COMPILE_TESTS=1`
   environment variable so the default `cargo test` run stays
   fast.

Run the end-to-end check explicitly:

```sh
MLPL_LOWER_RS_COMPILE_TESTS=1 cargo test -p mlpl-lower-rs
```

As later phases add more ops (arrays, matmul, reduce), extend
both suites: string-match for fast shape checks, one-or-two
end-to-end anchors per phase.
