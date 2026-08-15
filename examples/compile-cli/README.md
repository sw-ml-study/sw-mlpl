# Compile an MLPL Program to a Native Binary

This example shows the `mlpl build` CLI end to end: a tiny MLPL
source file becomes a self-contained native executable that links
only against `mlpl-rt` (no parser, no interpreter at run time).

For the full three-way comparison of the interpreter, the `mlpl!`
proc macro, and `mlpl build`, see `docs/compiling-mlpl.md`.

The quickest way to build and run any example is the wrapper scripts
in `examples/scripts/` (see `examples/scripts/README.md`):

```bash
examples/scripts/run.sh --native examples/compile-cli/sum-range.mlpl   # -> 21
```

The raw steps below show what that wrapper does.

## Files

- `sum-range.mlpl` -- a short MLPL program: build a range,
  reduce-sum it, print the scalar result (`range(7)` sums to 21).

## Build and run natively

This is a multi-workspace monorepo with no root `Cargo.toml`, so run
cargo from the `components/cli` workspace (the shared `.cargo/config.toml`
still writes the binary to the repo-root `target/`):

```bash
# Build the mlpl-build tool (first time only)
cargo build --release -p mlpl-build --manifest-path components/cli/Cargo.toml

# Compile the .mlpl file to a native binary (host target)
./target/release/mlpl-build examples/compile-cli/sum-range.mlpl -o /tmp/sum-range

# Run it
/tmp/sum-range
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
./target/release/mlpl-build \
    examples/compile-cli/sum-range.mlpl \
    --target wasm32-unknown-unknown \
    -o /tmp/sum-range.wasm
file /tmp/sum-range.wasm
# -> WebAssembly (wasm) binary module ...
```

### Running the wasm output

`wasm32-unknown-unknown` produces a **browser/embedding module**, not a
command-line program: it has no WASI, so there is no `_start` and no
stdout, and `wasmtime /tmp/sum-range.wasm` will not run it. It is meant to
be driven by a JavaScript host (the same way the web playground loads
its wasm bundle).

To run compiled MLPL as a standalone command-line wasm program, target
WASI and use a WASI runtime instead:

```bash
rustup target add wasm32-wasip1
./target/release/mlpl-build examples/compile-cli/sum-range.mlpl \
    --target wasm32-wasip1 -o /tmp/sum-range.wasm
wasmtime /tmp/sum-range.wasm
# -> 21
```

For native binaries (the common case) prefer
`examples/scripts/run.sh --native <file.mlpl>`.

## What the lowering supports

The compile path is narrower than the interpreter: it lowers the
ops and constructs needed for non-training code. As of v0.8.0 it
covers

- scalar and array literals, arithmetic, broadcasting
- `range`, `reshape`, `transpose`, `reduce_add` / `reduce_mul` with
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
