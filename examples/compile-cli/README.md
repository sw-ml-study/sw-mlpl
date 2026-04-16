# Compile an MLPL Program to a Native Binary

This example shows the `mlpl build` CLI end to end: a tiny MLPL
source file becomes a self-contained native executable that links
only against `mlpl-rt` (no parser, no interpreter at run time).

For the full three-way comparison of the interpreter, the `mlpl!`
proc macro, and `mlpl build`, see `docs/compiling-mlpl.md`.

## Files

- `hello.mlpl` -- a seven-line MLPL program: build a vector,
  reduce-sum it, print the scalar result.

## Build and run natively

From the repository root:

```bash
# Build the mlpl-build tool (first time only)
cargo build --release -p mlpl-build

# Compile the .mlpl file to a native binary
cargo run --release -p mlpl-build -- \
    examples/compile-cli/hello.mlpl -o /tmp/hello

# Run it
/tmp/hello
# -> 21
```

The generated binary is a regular Mach-O / ELF executable. It has
no dependency on `mlpl-repl` or `mlpl-eval`; the only MLPL crate
it links against is `mlpl-rt`.

## Cross-compile to WebAssembly

`mlpl build` forwards `--target <triple>` to cargo, so the same
source can be built for any target your Rust toolchain supports:

```bash
rustup target add wasm32-unknown-unknown
cargo run --release -p mlpl-build -- \
    examples/compile-cli/hello.mlpl \
    --target wasm32-unknown-unknown \
    -o /tmp/hello.wasm
file /tmp/hello.wasm
# -> WebAssembly (wasm) binary module ...
```

## What the lowering supports

The compile path is narrower than the interpreter: it lowers the
ops and constructs needed for non-training code. As of v0.8.0 it
covers

- scalar and array literals, arithmetic, broadcasting
- `iota`, `reshape`, `transpose`, `reduce_add` / `reduce_mul` with
  *positional* axis args
- `matmul` (with a static contraction check when both sides are
  labeled), `dot`
- element-wise math: `exp`, `log`, `sqrt`, `abs`, `pow`,
  `sigmoid`, `tanh_fn`
- label metadata (annotation syntax on assignment) and the
  corresponding label-propagation rules

And does **not** yet lower

- `repeat` / `train` / `for` loop bodies
- `param[shape]` / `tensor[shape]` / `grad(...)` (autograd needs
  tape state)
- `adam` / `momentum_sgd` and other optimizers
- the Model DSL (`chain`, `residual`, `linear`, `attention`,
  `causal_attention`, `embed`, ...)
- string-named axis reductions like `reduce_add(x, "seq")`
- `load` / `train_bpe` / `experiment` and other Saga 12+ surface

A program that uses any of those returns
`LowerError::Unsupported` at build time with a pointer to the
offending construct. Keep them in the interpreter (`mlpl-repl` or
the web REPL) for now; a future saga will extend the lowering.

## Under the hood

`mlpl-build` generates a tiny temporary cargo project whose
`main.rs` wraps the MLPL source in the `mlpl! { ... }` proc macro,
builds it via `cargo build --release`, and copies the resulting
binary to your requested output path. The `mlpl!` macro does the
actual lower-to-Rust translation at Rust compile time.

Because MLPL has no `exec(string)` primitive (no dynamic code
loading anywhere in the language), the compiled binary is closed:
the parser and evaluator are never linked in. See
`docs/compiling-mlpl.md` "Why this works" for the full argument.
