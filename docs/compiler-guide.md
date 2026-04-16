# Compiler Guide

How to get MLPL code out of the REPL and into a Rust program or
native binary. Partner doc to `docs/compiling-mlpl.md`, which
covers the *why* (design rationale, three-way comparison of
interpreter vs macro vs build); this one covers the *how* for
someone actually reaching for the compile path.

## When to reach for the compile path

- **`cargo run -p mlpl-repl`** -- authoring, exploring, teaching,
  inspecting shapes, iterating on a demo. The interpreter's
  `:vars` / `:describe` / `:trace` story only lives here. Edit
  and rerun are instant. This is the default.
- **`mlpl! { ... }` inside a Rust program** -- the MLPL is a
  small hot kernel inside a larger Rust app. You want the rest of
  cargo / rustc / the borrow checker to see it. Static label
  mismatches turn into `compile_error!` at rustc time.
- **`mlpl build foo.mlpl -o foo`** -- the MLPL *is* the program.
  You want a native binary with no runtime dependency on the
  interpreter or parser. Also the cross-compile story
  (`--target wasm32-unknown-unknown`, etc.).

Pick by *where the MLPL runs*, not by "which is faster" -- both
compile paths produce near-native Rust code.

## The `mlpl!` proc macro

Depend on the `mlpl` facade crate and use the `mlpl!` macro:

```toml
[dependencies]
mlpl = { path = "path/to/mlpl/crates/mlpl" }  # until crates.io
```

```rust
use mlpl::mlpl;

fn main() {
    let sum = mlpl! {
        v : [seq] = iota(5);
        reduce_add(v)
    };
    println!("{}", sum.data()[0]);  // 10
}
```

Rules of the macro:

- Statements inside `mlpl! { ... }` must be separated by `;`, not
  newlines. `proc_macro` tokenization strips REPL-style newlines.
- The macro result has type `mlpl_rt::DenseArray`. For a scalar
  result, `result.data()[0]` is the f64 value.
- Labels propagate through rustc as far as they are statically
  known. A matmul between `[seq, d]` and `[d_wrong, h]` becomes a
  `compile_error!` before any code runs.
- Everything that runs through the macro also runs through the
  interpreter; the parity harness
  (`crates/mlpl-parity-tests/tests/parity_tests.rs`) asserts
  bit-equal output on deterministic ops.

## The `mlpl build` subcommand

A `.mlpl` script becomes a native binary:

```bash
cargo build --release -p mlpl-build     # build the tool itself

# Compile a script
cargo run --release -p mlpl-build -- examples/compile-cli/hello.mlpl -o /tmp/hello
/tmp/hello
# -> 21

# Cross-compile for WASM
rustup target add wasm32-unknown-unknown
cargo run --release -p mlpl-build -- examples/compile-cli/hello.mlpl \
    --target wasm32-unknown-unknown -o /tmp/hello.wasm
```

What happens under the hood:

1. `mlpl-build` reads the `.mlpl` source.
2. `mlpl_parser::lex` + `mlpl_parser::parse` + `mlpl_lower_rs::lower`
   run eagerly. A syntax error, parse error, or static
   `LowerError` (e.g. an unsupported construct or a static label
   mismatch) fails here, before cargo is spun up. Errors are
   prefixed `mlpl-build: ...` for easy grepping.
3. A tiny temp cargo project is generated. Its `main.rs` wraps
   the lowered MLPL in `mlpl::mlpl! { ... }` and prints the scalar
   result.
4. `cargo build --release` (with `--target <triple>` forwarded
   when asked).
5. The resulting binary is copied to your `-o` path.

The binary links only against `mlpl-rt` + its transitive
dependencies (`mlpl-array`, `mlpl-core`). It does not carry a
parser, an interpreter, or any of the Saga 11+ runtime -- see
`docs/compiling-mlpl.md` "Why this works" for why the language's
closed surface makes this natural.

## What the compile path lowers today

**Lowered** (`cargo bench -p mlpl-bench` runs the full set; see
also `examples/compile-cli/hello.mlpl` and the curated corpus in
`crates/mlpl-parity-tests/tests/parity_tests.rs::PARITY_CASES`):

- scalar and array literals; `+`, `-`, `*`, `/`, unary `-`;
  broadcasting between a scalar and an array
- `iota(n)`, `reshape(a, dims)`, `transpose(a)`
- `reduce_add` / `reduce_mul` with a *positional* axis int
  (`reduce_add(x, 0)`, not `reduce_add(x, "feat")`)
- `matmul(a, b)`, `dot(a, b)`
- element-wise math: `exp`, `log`, `sqrt`, `abs`, `pow`, `sigmoid`,
  `tanh_fn`
- labeled shapes: annotation syntax (`x : [batch, feat] = ...`),
  label propagation through elementwise / matmul / reduction

**Not lowered today** (returns `LowerError::Unsupported` at build
time; keep in the interpreter):

- `repeat N { ... }`, `train N { ... }`, `for row in X { ... }`
- `param[shape]`, `tensor[shape]`, `grad(expr, wrt)` (all require
  tape state at runtime)
- `adam` / `momentum_sgd` / `cosine_schedule` /
  `linear_warmup` / any optimizer surface
- the entire Model DSL: `chain`, `residual`, `linear`,
  `tanh_layer` / `relu_layer` / `softmax_layer`, `rms_norm`,
  `attention`, `causal_attention`, `embed`,
  `sinusoidal_encoding`, `apply`, `params`
- string-named axis arguments like `reduce_add(x, "seq")` (even
  though the labeled-axis *propagation* is lowered)
- Saga 12 surface: `load`, `load_preloaded`, `shuffle`, `batch`,
  `split`, `val_split`, `tokenize_bytes`, `train_bpe`,
  `apply_tokenizer`, `decode`, `experiment "..." { ... }`,
  `compare`, `:experiments` REPL command
- Saga 13 LM surface: `cross_entropy`, `sample`, `top_k`,
  `last_row`, `concat`, `attention_weights`, `shift_pairs_x/y`
- all `:`-prefixed REPL commands (they aren't in the grammar; the
  lexer errors on `:`)

## Troubleshooting

**"unexpected token ':' at 0..1"** -- you have a `:`-prefixed
REPL command in your `.mlpl` file. `mlpl build` compiles MLPL
*expressions*, not REPL commands. Remove the `:` lines.

**"LowerError::Unsupported: ..."** -- you're using something in
the "Not lowered today" list above. Keep the program in the
interpreter, or rewrite the offending piece to use only the
lowered subset.

**"LowerError::StaticShapeMismatch: ..."** -- a matmul between
two labeled operands whose contraction axes don't agree. This is
a feature: the labels caught the bug before the code ran. Fix
the shapes.

**"unknown function: <foo>"** -- either a typo, or a builtin the
compile path hasn't wired up yet. Check
`crates/mlpl-lower-rs/src/lib.rs` to see the supported names.

## Speed

See `docs/benchmarks.md`. Six workloads, reproducible via
`cargo bench -p mlpl-bench`. Compiled is 1.6x to 6.6x faster than
the interpreter on the author's laptop, depending on workload.
The win is largest on per-op-dispatch-bound workloads and
smallest on workloads dominated by memory traffic inside
identical inner kernels.

## Where to look next

- `docs/compiling-mlpl.md` -- the design doc (three-way
  comparison, non-goals, the closed-surface argument)
- `docs/milestone-compile-to-rust.md` -- the original saga
  retrospective
- `examples/compile-cli/` -- a runnable end-to-end example
- `crates/mlpl-lower-rs/src/lib.rs` -- the lowering itself
- `crates/mlpl-rt/src/lib.rs` -- the runtime target that
  compiled programs call into
- `crates/mlpl-parity-tests/tests/parity_tests.rs` -- the parity
  gate that keeps the two paths honest
