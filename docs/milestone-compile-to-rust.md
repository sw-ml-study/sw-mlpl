# Compile-to-Rust Milestone (COMPLETE, v0.8.0)

Status: **shipped** as `v0.8.0-compile-rs` on 2026-04-15. This
document was originally written as a design-direction scout and
has since been executed. The design outlined below held up in
practice; the retrospective at the end records what actually
shipped vs. what was deferred, and the measured 9x speedup from
the parity harness.

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

## Retrospective (v0.8.0-compile-rs)

### What shipped

Eight steps in the agentrail saga, each under its own commit.
The design held up: one codegen crate, three targets, no JIT,
no LLVM, emit Rust and let rustc do lowering.

| Step | Slug | What landed |
|---|---|---|
| 001 | `mlpl-rt-scaffolding` | Runtime target crate, typed wrappers around `DenseArray`/`LabeledShape`, 7 primitive fns |
| 002 | `lower-rs-scalar` | Scalar-arithmetic codegen, end-to-end rustc-compile-and-run test |
| 003 | `lower-rs-arrays` | Array literals, variable bindings, 6 phase-1 fncalls, `array_lit` runtime primitive |
| 004 | `lower-rs-labels` | `label` / `relabel` / `reshape_labeled` / annotation syntax; `matmul` with static contraction check; `StaticShapeMismatch` error |
| 005 | `mlpl-proc-macro` | `mlpl-macro` (proc-macro) + `mlpl` facade crate with hidden `__rt` re-export; static label mismatches become `compile_error!` |
| 006 | `interpreter-compiler-parity` | `mlpl-parity-tests` harness: 9 programs agree bit-for-bit; speedup measurement |
| 008 | `mlpl-build-subcommand` | `mlpl-build foo.mlpl -o bin [--target <triple>]` binary: eager fail on parse/lower error, cargo shell-out, wasm cross-compile verified |
| 007 | `compile-to-rust-release` | This step: version bump 0.7.5 -> 0.8.0, banners, saga/status/plan docs, tag `v0.8.0-compile-rs` |

Step numbering note: during saga init a shell-quoting mishap
placed step 008 (build subcommand) after step 007 (release) in
the agentrail numeric slots. The documented execution order
swapped the two at runtime, and commit messages record which
slot ran which step's content.

### Measured speedup

The parity harness's `compiled_speedup_measurement` test times a
100x100 reshape + row-reduce + column-reduce + sum workload in
both modes, median of 5:

    interpreter = 479,500 ns
    compiled    =  53,000 ns
    ratio       = 9.05x

Expected ballpark was 2-10x per the milestone doc's phase-7
sketch; 9x is at the high end for an array-heavy workload.
Scalar-heavy code would show a larger ratio; fat-vector code a
smaller one.

### What was deferred

`TensorCtor` (`param[...]`, `tensor[...]`), `Repeat`, `Train`,
autograd (`grad`), optimizers (`adam`, `momentum_sgd`), and the
Model DSL (`chain`, `linear`, activations, `residual`, `apply`)
are all out of compile scope. They need either (a) tape-state
lowering (autograd + optimizers), (b) control-flow lowering
(Repeat / Train), or (c) apply-time dispatch lowering (Model
DSL). Each is a real piece of design work and would expand the
saga meaningfully. Leaving them for a follow-up keeps the v0.8
release small enough to ship cleanly.

Consequence: `mlpl-build` and `mlpl!` today work for the
numerical-expression subset of MLPL. Training loops stay in the
interpreter; once the user has a trained model, serializing its
parameters and calling a compiled forward pass via `mlpl-build`
is feasible but not yet wired up.

### Design decisions that held up

- **Emit Rust, not LLVM.** Every SIMD/BLAS/Rayon/cross-compile
  win comes free. The `mlpl-lower-rs` crate is ~300 LOC of
  `quote!` and a match arm per primitive.
- **`DenseArray`-everywhere return type.** Every lowered
  expression is a `DenseArray`. `from_scalar` wraps literals;
  `apply_binop` threads through BinOps; `map` handles unary neg.
  That uniform shape meant phase 2 added array literals and
  fncalls without rewriting phases 1's scalar path.
- **No `exec(string)`.** MLPL still has no runtime code-loading
  primitive, which is why compiled binaries don't need to ship
  a parser. This constraint must be preserved going forward.
- **Path-configurable runtime.** `LowerConfig { rt_path }`
  lets the proc macro emit `::mlpl::__rt::...` (via the facade)
  while direct lower users emit `::mlpl_rt::...`. Same
  TokenStream producer; two target paths.

### Design surprises

- **Proc-macro newline handling.** `TokenStream::to_string()`
  inserts `\n` in some group formattings, which collided with
  the MLPL parser's use of `\n` as a statement separator. Fix:
  macro expand() rewrites every `\n` to a space before lexing;
  users inside `mlpl! { ... }` must separate statements with `;`.
  Documented; not a big deal once known.
- **Static label tracking only needed a small env.** The
  `Ctx::known_labels` HashMap plus the four cases in `labels_of`
  (Ident / label+relabel / reshape_labeled / transpose) is
  enough to catch the realistic `matmul(a, transpose(b))` kind
  of mismatch. A bigger static analyzer would be overkill.

### Follow-up candidates

Ordered roughly by payoff:

1. Lower Model DSL (`chain`/`linear`/activations/`residual`).
   Probably the single biggest user-facing win -- it would let
   `mlpl-build` ship a trained model's forward pass as a native
   binary. Requires apply-time dispatch lowering but no
   autograd.
2. Lower `Repeat` / `Train`. Straight-line rust `for` loops
   with `step` binding. Opens up compile for any loop-bodied
   MLPL program, not just expressions.
3. Span-preserved `compile_error!`. Today errors point at the
   macro call site via `Span::call_site()`; mapping back to
   individual MLPL tokens inside the macro input would give
   rustc-style carats at the offending op.
4. Lower autograd + optimizers. Requires representing the tape
   at compile time or emitting runtime tape calls. Worth it
   only after the Model DSL path exists so there's something to
   differentiate.

None of these are on the Saga 12 roadmap; they live in a
future "compile-to-rust v2" saga that would extend the existing
`mlpl-lower-rs` crate rather than rewrite it.
