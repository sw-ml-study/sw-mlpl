# Compile-to-Rust Milestone (FUTURE SAGA, exploratory)

Status: **design direction, not yet scheduled**. This document
captures the direction so a future saga can be planned from it.

## Why this exists

MLPL today is REPL-first: parser + AST-walking evaluator + runtime
built on `ndarray`/`DenseArray`. That is the right core for an
APL-workspace experience, but it constrains two use cases:

1. **Standalone binaries.** Users cannot take a `.mlpl` script and
   ship a native executable. WASM exists (the live demo on GitHub
   Pages), but WASM has real scaling limits (threading story,
   SIMD constraints, no direct GPU, memory ceilings). Users who
   want to run MLPL code on Mac or Linux at non-toy scale need a
   native binary path.
2. **Embedding in Rust applications.** A Rust codebase today
   cannot drop MLPL notation into hot code and get static dispatch.
   It has to either shell out to the REPL or call into the
   interpreter crates and pay interpretation overhead.

Both problems resolve cleanly if MLPL can lower to **Rust source**
and let the host toolchain (rustc + cargo + LLVM) do the heavy
lifting: SIMD, BLAS, Rayon, cross-compilation, debuggers, Metal
and CUDA via `wgpu`/`candle`.

## Key enabling fact

MLPL has **no `⍎`-equivalent**. There is no `exec(string)`, no
`eval(string)`, no string-to-code reflection primitive anywhere in
the builtin surface (see `crates/mlpl-runtime/src/builtins.rs`).
REPL introspection commands (`:vars`, `:describe`, `:builtins`)
are read-only. Compiled MLPL programs therefore do **not** need to
ship a parser or interpreter in the runtime binary. This is the
single most important constraint that makes AOT viable, and it
must be preserved -- adding `exec(string)` later would break the
compile path.

## Design direction

### Shape: hybrid, one codegen backend, three targets

```
MLPL source  ---+
                |
                v
        +------------------+
        |   mlpl-parser    |  (already exists)
        +------------------+
                |
                v
        +------------------+
        |  mlpl-lower-rs   |  (NEW: MLPL AST -> Rust TokenStream)
        +------------------+
                |
        +-------+---------+-----------+
        |                 |           |
        v                 v           v
  proc macro         cargo build   wasm32 target
  `mlpl! { ... }`    native bin    web artifact
```

One lowering crate drives all three. The interpreter stays unchanged
for REPL / exploration.

### Target 1: `mlpl!` proc macro

```rust
use mlpl::mlpl;

fn main() {
    let r = mlpl! {
        +/ (iota 100) * 2
    };
    println!("{}", r);
}
```

- Proc-macro crate imports `mlpl-parser`, walks the AST, emits a
  `TokenStream` that calls into `mlpl-rt` (a thin runtime built on
  `ndarray` or `DenseArray` directly).
- Shapes become const generics where statically known; runtime
  shapes otherwise.
- **LabeledShape (Saga 11.5 deliverable) is a compile-time gift**:
  named axes resolve einsum-class operations at macro expansion
  rather than runtime dispatch.
- Errors use span-preserved `compile_error!` so rustc points carats
  at the MLPL source inside the macro.

Prior art proving this is tractable: `cxx::cxx!`, `nalgebra::matrix!`,
`ndarray::s!`, Cranelift's ISLE. Complex parse + codegen inside
proc macros is a solved shape.

### Target 2: `mlpl build foo.mlpl` -> native binary

- New subcommand on the `mlpl` CLI (or a separate `mlpl-build` binary).
- Generates a tiny cargo project in a temp dir:

  ```rust
  // generated main.rs
  use mlpl::mlpl;
  fn main() {
      mlpl! { /* user's .mlpl source inlined */ }
  }
  ```

- Shells out to `cargo build --release`.
- Output: native binary at the user's requested path.
- Cross-compilation falls out of cargo for free (`--target aarch64-apple-darwin`,
  `--target x86_64-unknown-linux-gnu`, etc.).

### Target 3: WASM (already exists, newly leveraged)

The same lowering pipeline, compiled with `--target wasm32-unknown-unknown`,
produces the web artifact. Today's web build uses the interpreter;
a compiled web build would be smaller and faster for fixed demos
(at the cost of losing the REPL in the page -- keep both paths).

## Non-goals

- **A JIT.** The per-primitive dispatch overhead in an array
  language is already amortized across array size. JIT complicates
  debugging, obscures the trace/viz story, and is not the bottleneck
  for whole-array workloads. Revisit only if profiling shows
  scalar-loop interpretation dominates something real.
- **LLVM or Cranelift as first-party backends.** Emitting Rust and
  letting rustc lower to LLVM is strictly more leveraged: we inherit
  SIMD, BLAS, Rayon, thread-safety guarantees, Metal/CUDA via
  `wgpu`/`candle`, and the entire cargo/rustup ecosystem.
- **Dynamic MLPL features that would force an embedded parser.**
  No `exec(string)`, no dynamic `require`, no runtime macro expansion.
  These would violate the core constraint that makes AOT viable.
- **REPL parity inside the macro.** The `mlpl!` macro is for static
  code. REPL / workspace features (inspection, redefinition,
  `:describe`, etc.) stay in the interpreter. Two modes, same
  surface syntax -- this is the historical APL pattern (explore in
  workspace, ship in static language).

## Phases (rough sketch, to be refined when saga is scheduled)

1. **`mlpl-rt` scaffolding.** Extract a minimal runtime crate that
   the generated code will call into. Thin wrappers around
   `DenseArray` or `ndarray`. No builtins yet -- just the value
   type, shape, labeled shape, and a handful of primitive ops.
2. **`mlpl-lower-rs` crate.** Walks `mlpl-parser` AST and emits
   Rust `TokenStream`. Starts with scalar arithmetic, extends op
   by op. Each primitive is independently addable.
3. **`mlpl!` proc macro.** Thin wrapper over `mlpl-lower-rs`.
   Span-preserved error reporting.
4. **Shape inference pass.** Static shapes become const generics;
   labeled axes resolve at macro-expansion time.
5. **`mlpl build foo.mlpl` subcommand.** Generate cargo project,
   shell to rustc, emit binary.
6. **Parity tests.** Every `mlpl-eval` test that runs a pure
   expression gets mirrored as a compile-and-run test. Interpreter
   and compiler must agree.
7. **Demo rewrites.** Convert one or two on-disk demos (e.g. the
   moons classifier, the transformer block) to native binaries.
   Compare runtime against the interpreter baseline.
8. **Docs + release.** Tutorial lesson on the macro; release tag.

## Risks and open questions

- **`DenseArray` vs `ndarray` in `mlpl-rt`.** The interpreter uses
  our own `DenseArray`. For the compile path, using `ndarray`
  directly gets us BLAS and Rayon integration for free. Option A:
  `mlpl-rt` re-exports `DenseArray` and we grow its capabilities.
  Option B: `mlpl-rt` is an `ndarray` facade and we port semantics.
  Decide before phase 1.
- **Random number generation.** `randn`, `rand`, etc. need a seeded
  RNG story that works at both interpret time and compile time.
  Likely: emit calls into `mlpl-rt::prng` which wraps the same
  `rand`/`rand_distr` crates the interpreter already uses.
- **Model DSL lowering.** `chain(...)`, `residual(...)`, `train { }`
  are the most complex constructs in the language. They rely on
  autograd tape state. Compiling them requires either (a) running
  the tape at compile time and emitting fused forward+backward
  code, or (b) emitting runtime tape operations. (b) is easier,
  (a) is faster. Start with (b); revisit (a) as an optimization.
- **Error message quality.** Span-preserved `compile_error!` is
  table stakes; getting the message text as good as the REPL's
  `EvalError` rendering will take deliberate work in phase 3.
- **`mlpl-rt` size budget.** If `mlpl-rt` pulls in `ndarray` +
  `rand` + `ndarray-linalg` + BLAS, the minimum native binary
  will be several MB. That is fine for binaries, potentially
  awkward for WASM. Consider feature flags from day one.

## Dependencies on earlier sagas

- **Saga 11.5 (Named Axes) must ship first.** LabeledShape is the
  compile-time key that makes einsum-class operations resolvable
  at macro expansion. Without it, every matmul / reduce chain
  stays dynamic and loses most of the compile-time win.
- **Saga 11 (Model DSL)** defines the tree structure that the
  compiler has to lower. `ModelSpec` is already compile-friendly
  (it's a finite tree of layer variants), so this saga does not
  need additions -- just a lowering.

## Hand-off note for the future saga planner

- Do **not** add `exec(string)` or any other runtime-code primitive
  between now and this saga, however convenient it seems. The
  compile path depends on its absence.
- Keep `mlpl-parser` host-toolchain-friendly: no `std::fs` inside
  the parser, nothing that would fail when called from inside a
  proc macro. It is already clean today; keep it that way.
- When Saga 11.5 step 007 exports labels through the trace JSON,
  think of that format as a preview of what `mlpl-lower-rs` will
  consume -- the shapes with labels are exactly the metadata the
  codegen will need.
