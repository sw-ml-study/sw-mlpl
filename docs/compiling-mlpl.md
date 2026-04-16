# Compiling MLPL

**Status:** shipped in v0.8.0. This doc is user-facing: it
describes what each mode *is for*, shows examples, and lays out
the trade-offs so you can pick the right tool for your situation.
The design rationale -- why this path rather than a JIT or LLVM
backend -- and the retrospective on the saga live in
`docs/milestone-compile-to-rust.md`. See
`examples/compile-cli/` for a runnable `mlpl build` walkthrough.

## Three ways to run MLPL

MLPL today ships one runtime (the interpreter). The compile-to-rust
saga adds two more. All three share a parser, evaluator semantics,
and label/shape story; they differ in *when* the compilation step
happens and *what* ships at runtime.

| Mode | When compiled | Runtime deps | Startup | Hot-path speed | REPL | Edit-run cycle |
|------|---------------|--------------|---------|----------------|------|----------------|
| Interpreter (REPL) | Never -- parses + walks AST each statement | Full `mlpl-eval` + `mlpl-runtime` | Instant | Slow per-op; fast per-array | Yes | Lightning |
| `mlpl!` macro | At Rust compile time | Only `mlpl-rt` | Whatever your Rust binary's is | Near-native Rust | No | Rust compile time |
| `mlpl build foo.mlpl` | Once, ahead of time | Only `mlpl-rt` | Very fast (stripped binary) | Near-native Rust | No | rustc compile + run |

The interpreter stays the default for exploration and teaching.
The two compile modes exist to move MLPL out of the REPL and into
production-shaped code: embedded in a Rust program, or standing
alone as a native executable.

## Use cases

### Interpreter (REPL)
- **Learning / tutorials.** The web REPL at
  [sw-ml-study.github.io/sw-mlpl](https://sw-ml-study.github.io/sw-mlpl/)
  is the starting surface.
- **Exploration.** Try a change, see the result, try another. The
  interpreter's `:vars` / `:describe` inspection matters here.
- **Prototyping shapes and labels.** Saga 11.5 labeled shapes pay
  off most during interactive session: label mismatches fail fast,
  `:describe X` shows `X: [seq=6, d_model=4]`, errors point at the
  offending op with both shapes rendered.

### `mlpl!` proc macro (step 005, v0.8.0)
- **Embedding MLPL inside a Rust application.** When the rest of
  your app is Rust but a hot inner kernel is nicest expressed in
  array notation.
- **Config-as-code for model definitions.** A layer spec written
  in MLPL gets checked at Rust compile time rather than first run.
- **Unit tests of MLPL fragments from Rust.** Write a test in Rust
  that asserts on the numeric output of a small MLPL expression,
  with the expression type-checked by rustc.

```rust
use mlpl::mlpl;

fn main() {
    let result = mlpl! {
        x : [seq] = iota(5)
        reduce_add(x, "seq")
    };
    println!("sum = {}", result.data()[0]);  // sum = 10
}
```

### `mlpl build` native binary (step 008, v0.8.0)
- **Ship a script as a stand-alone tool.** A small MLPL program
  becomes a native Mac/Linux executable with no runtime.
- **Cross-compile for other targets.** Uses cargo's `--target`
  flag, so the same MLPL source builds for x86_64-linux,
  aarch64-darwin, wasm32, etc.
- **CI jobs and batch pipelines.** Anywhere startup time matters
  or the Rust/MLPL boundary is a hassle.

```sh
mlpl build demos/kmeans.mlpl -o kmeans
./kmeans                                  # native, no runtime
mlpl build demos/kmeans.mlpl --target wasm32-unknown-unknown \
    -o kmeans.wasm                        # cross-compile
```

## Why this works: MLPL has no `exec(string)`

APL has an "execute" primitive (the "Quad-LeftTack" glyph) that
takes a string and evaluates it as code. That forces every compiled APL binary to ship a full parser
and interpreter in its runtime -- a huge tax. MLPL does **not**
have this escape hatch. The builtin surface (see
`crates/mlpl-runtime/src/builtins.rs`) is closed: no `exec`, no
`eval`, no string-to-code reflection anywhere. REPL introspection
(`:vars`, `:describe`) is read-only.

Because of that, compiled MLPL binaries link only against
`mlpl-rt` (array primitives) and never need the parser or
evaluator at run time. This is the single property that makes AOT
compilation to Rust natural and small.

Preserving this property is a hard constraint going forward. See
`docs/milestone-compile-to-rust.md` "Non-goals" for the list of
features that would break it and must never ship.

## Same semantics, two runtimes

A key guarantee: the compiled output of an MLPL program produces
the same numeric results as the interpreter. The parity test
harness (step 007) iterates a curated subset of interpreter tests
through both paths and asserts bit-equal output for deterministic
ops. If parity breaks, the compiler is wrong; the interpreter is
the source of truth.

Labels, shapes, error semantics all carry through:

- A matmul with a static label mismatch becomes a rustc
  `compile_error!` (compile-time) in `mlpl!`, or a structured
  `ShapeMismatch` at run time in `mlpl build` binaries -- same
  error information either way.
- `transpose` swaps labels, `reduce_add("feat")` drops that
  axis's label, `map()` preserves labels -- identical to the
  interpreter.

## Pros and cons

### `mlpl!` proc macro

**Pros**
- Zero runtime: compiles out to straight Rust calls into
  `mlpl-rt`.
- Errors point at MLPL source with rustc carats (span-preserved
  `syn::Error` -> `compile_error!`).
- Static label checks: mismatches that would be runtime errors in
  the REPL become compile errors inside the macro where both
  operands' labels are statically known.
- Composes with the rest of your Rust code -- borrow checker,
  types, cargo, cross-compilation all Just Work.

**Cons**
- Rust compile times apply to every change. Fine for production
  code, wrong tool for iterative exploration (use the REPL).
- No dynamic features: no runtime `exec`, no code loading, no
  MLPL-level metaprogramming. This is the design constraint that
  makes the compile path work.
- Debugging the generated Rust is indirect -- `cargo expand`
  shows what your `mlpl!` block became; plan to use it.

### `mlpl build foo.mlpl`

**Pros**
- Produces a real native binary. No MLPL runtime ships, no JIT
  warmup, startup is just the OS loading an ELF or Mach-O.
- Cross-compilation comes free via cargo's `--target`.
- Binary size stays small: only `mlpl-rt` + its transitive
  dependencies (`mlpl-array`, `mlpl-core`, probably `ndarray`
  later) get linked, not the parser or interpreter.
- Deployment story is the same as any Rust binary.

**Cons**
- Each build invokes `cargo` + `rustc` under the hood: seconds,
  not milliseconds. Fine for deployment, wrong for edit-run loops.
- Requires a Rust toolchain on the build machine. Target machine
  needs nothing -- the binary is self-contained.
- Debugging gets further from the MLPL source; interpreter is
  still the right surface for "why did this go wrong".

### Interpreter (reference point)

**Pros**
- Instant feedback, introspection (`:vars`, `:describe`,
  `:builtins`), inline SVG rendering in the web REPL, tutorial
  lesson flow.
- Full language surface, including any future features the
  compile path doesn't cover yet.
- Structured errors with labeled shapes land here first
  (`EvalError::ShapeMismatch`).

**Cons**
- Per-op dispatch overhead. Acceptable because array ops are fat
  (amortize across elements), but scalar-heavy code pays for it.
- Can't ship as a stand-alone binary without shipping the
  interpreter.

## How to choose

- **Writing, exploring, or teaching?** Interpreter.
- **Small hot path embedded in a larger Rust app?** `mlpl!` macro.
- **A script that should become an executable?** `mlpl build`.
- **Not sure?** Start in the REPL. Once the code works, either
  freeze it into a `mlpl!` block inside your Rust project, or
  build it with `mlpl build`. The REPL is the authoring surface;
  the compile modes are ways to ship.

## Status and what shipped (v0.8.0)

- **`mlpl-rt`** runtime target -- typed primitive wrappers
  around `DenseArray` / `LabeledShape`. This is what compiled
  MLPL programs link against at runtime; no parser, no
  evaluator.
- **`mlpl-lower-rs`** lowers the AST to a Rust `TokenStream`.
  Scalar + array + label lowering, plus static matmul
  contraction checks: when both operands' labels are known at
  lower time, a mismatch is caught before rustc runs.
- **`mlpl!` proc macro** (through the `mlpl` facade crate) --
  embed MLPL inside a Rust program. Span-preserved errors land
  in rustc as `compile_error!` with MLPL-source carats.
- **`mlpl build foo.mlpl -o bin`** -- compile a `.mlpl` file to
  a native binary via cargo + rustc. Cross-compilation comes
  free via `--target <triple>`; native and
  `wasm32-unknown-unknown` are the tested paths.
- **Parity harness** -- nine curated programs run through both
  the interpreter and the compile path with bit-equal output
  assertions on deterministic ops. Measured 9x speedup on a
  100x100 reshape+reduce workload (interpreter 479us ->
  compiled 53us, median of 5).

- **Usage note for the macro.** Inside `mlpl! { ... }`,
  statements must be separated by `;` because `proc_macro`
  tokenization strips REPL-style newlines. Label annotations,
  multi-statement programs, and static label checks all work
  through the macro path -- a matmul with mismatched
  contraction labels becomes a `compile_error!` at build time
  rather than a runtime error.

- **Out of scope (deferred).** `TensorCtor` (`param` /
  `tensor`), `repeat` / `train` / `for` loop bodies, autograd
  (`grad`), optimizers (`adam` / `momentum_sgd`), and the Model
  DSL (`chain` / `linear` / activations / `embed` /
  `causal_attention`) are not lowered yet -- they require tape
  state or loop lowering. Programs that use them return
  `LowerError::Unsupported` from the compile path; the
  interpreter is the only way to run them today. A future saga
  will extend the lowering.

- **Not planned:** JIT, LLVM direct backend, an
  `exec(string)` primitive. See
  `docs/milestone-compile-to-rust.md` "Non-goals".

For a concrete `mlpl build` walkthrough see
`examples/compile-cli/README.md`.
