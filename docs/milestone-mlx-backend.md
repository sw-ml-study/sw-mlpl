# MLX Backend Milestone (Saga 14, v0.11.0)

## Why this exists

Saga 13 proved the platform thesis on CPU: a few lines of MLPL
train a tiny transformer LM, generate text, and render its own
attention. The next bottleneck is speed. The Saga 13 Tiny LM
trains at interpreter pace on an M-class laptop; an accelerator
backend is the proper lever.

Apple Silicon is this project's baseline dev hardware, and MLX
is its native array framework: lazy graph execution, automatic
operator fusion, unified CPU+GPU memory, and no kernel-per-device
dance. Its array semantics already agree with MLPL's
`DenseArray` (row-major, shape-labeled, consistent broadcast
rules). Saga 14 introduces MLX as MLPL's first accelerator
backend behind the same source surface: a program that trains on
CPU should train on MLX without source changes.

CUDA (Saga 17) is the second accelerator backend. The MLPL
source surface shipped in Saga 14 is the same surface CUDA will
land behind. See `docs/using-mlx.md` for the user-facing design
sketch; this doc is the implementation plan.

## Non-goals

- **No CUDA.** Saga 17.
- **No distributed execution.** Saga 17.
- **No LoRA / quantization.** Saga 15.
- **No new Model DSL layers.** The existing `linear`, `chain`,
  `residual`, `rms_norm`, `attention`, `causal_attention`,
  `embed`, `sinusoidal_encoding`, activations, and
  `cross_entropy` ship on MLX as-is.
- **No checkpoint format.** Still deferred to Saga 15+.
- **No pretraining claims.** Target is the Saga 13 Tiny LM demo
  running faster, not a bigger model.

## Quality requirements (every step)

Identical to Saga 13:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` changes are committed whenever they change.
7. MLX-feature tests are gated so CI on non-Apple-Silicon hosts
   still passes (`#[cfg(all(target_os = "macos",
   target_arch = "aarch64", feature = "mlx"))]` or the closest
   equivalent supported by `mlx-rs`). Where MLX is unavailable,
   the CPU path remains authoritative.

## What already exists

- `mlpl-rt` runtime target (scalar, array, fncall, variable,
  labels, matmul with static contraction check) used by both
  `mlpl!` proc macro and `mlpl-build` CLI.
- `mlpl-parity-tests` harness proving bit-for-bit agreement
  between interpreter and compile-to-Rust paths on 9 programs.
- `mlpl-bench` Criterion harness comparing interpreter vs
  compiled MLPL (9.05x measured speedup on 100x100 reshape+
  reduce).
- `mlpl-lower-rs` path-configurable AST -> `TokenStream` codegen
  (the `runtime_prefix` knob makes swapping runtimes cheap).
- Full Saga 11/13 Model DSL, autograd, optimizers, and the
  Tiny LM end-to-end demo (`demos/tiny_lm.mlpl`).
- Labeled axes and structured `EvalError::ShapeMismatch`
  propagate through every existing op.

## Phase 1 -- mlpl-mlx runtime target (3 steps)

### Step 001 -- mlpl-mlx crate skeleton + first primitive
New `crates/mlpl-mlx` sibling to `mlpl-rt`. `mlx-rs` dependency
behind a `mlx` Cargo feature. Ports exactly one primitive --
`matmul(a, b)` -- backed by MLX's `matmul` and returning an
MLX array wrapped in the same `DenseArray` / `LabeledShape`
shape as `mlpl-rt`. Parity test vs `mlpl-rt::matmul` on a
`[8, 4] @ [4, 8]` fixture (bit-for-bit or within a documented
fp32 tolerance; decide during TDD). `cargo test -p mlpl-mlx
--features mlx` gates the test; default build stays green on
non-Apple hosts.

### Step 002 -- core elementwise + shape ops on MLX
Port the shape- and element-level primitives the existing
interpreter uses most: `add`/`sub`/`mul`/`div`/`neg`, `exp`/
`log`/`tanh`/`sigmoid`/`relu`, `reshape`, `transpose`. Each
primitive lands with a unit test + parity test vs `mlpl-rt`.
Labels propagate identically -- the MLX runtime does not own
`LabeledShape`, it borrows `mlpl-core`'s. Gradcheck is not in
scope yet (step 006).

### Step 003 -- reductions + softmax + log-softmax on MLX
`reduce_add`/`reduce_mul`/`mean`/`argmax` (axis-aware, label-
aware), `softmax`, `log_softmax`, and `cross_entropy`.
Numerical-stability invariants from the CPU path (max-
subtraction log-softmax) carry over. Parity tests on `[4, 3]`
and `[2, 3, 5]` fixtures. At the end of this phase the MLX
runtime can compute forward passes of every primitive the Tiny
LM uses.

## Phase 2 -- Device placement + backend dispatch (2 steps)

### Step 004 -- device("...") scoped form + parser surface
New `device("mlx") { body }` scoped form in the parser/AST
(`Expr::Device`), mirroring `experiment`'s shape. Inside the
block, evaluator dispatches array allocations and ops through
the MLX runtime when the `mlx` feature is enabled; falls back
to CPU (with a one-time warning) otherwise. Labels and shapes
propagate identically across the boundary. Tests: parser
round-trip, evaluator returns the same shape+labels as the CPU
path, `device("cpu") { }` is a no-op and works on every host.

### Step 005 -- backend dispatch in the Model DSL
`apply(model, X)` and `params(model)` route through the active
backend when `X` was allocated inside a `device("mlx")` block.
A model owns its parameters on one device -- moving between
devices requires an explicit `to_device("cpu")` / `to_device
("mlx")` builtin (single-direction helpers that copy tensors;
tests cover round-trip equality). The Saga 11 models
(`linear`, `chain`, `residual`, `rms_norm`, `attention`,
`causal_attention`) produce bit-for-bit (or tolerance-bounded)
equivalent outputs on both devices for fixed seeds.

## Phase 3 -- Autograd on MLX (2 steps)

### Step 006 -- tape-lowered ops on MLX + gradcheck parity
`grad(expr, wrt)` works inside a `device("mlx")` block. The
autograd tape's primitives (`add`, `mul`, `matmul`, `exp`,
`log`, `relu`, `tanh`, `sigmoid`, `softmax`, `sum`, `mean`,
`transpose`, `reshape`) all ship MLX-backed backward paths,
each gradcheck-verified against finite differences on fixtures
matching Saga 9's. Where MLX's own `vjp` / `grad` helps,
`mlpl-mlx` is free to use it; where it doesn't, we hand-write
the backward. Either way, the invariant is that
`grad(expr, wrt)` on MLX matches the CPU path to within a
documented tolerance.

### Step 007 -- optimizers + train { } on MLX
`adam`, `momentum_sgd`, `train N { body }`, `last_losses`,
`experiment "name" { }`, and the `_metric` capture all work
unchanged inside `device("mlx") { }`. The `OptimizerState` map
holds MLX-typed tensors when the surrounding scope is MLX.
Parity test: one Adam step on the Saga 11 `tiny_mlp` demo
produces the same parameter update (within tolerance) on CPU
and MLX.

## Phase 4 -- Tiny LM on MLX + benchmarks (1 step)

### Step 008 -- tiny LM MLX variant + bench harness
`demos/tiny_lm_mlx.mlpl` wraps the Saga 13 `demos/tiny_lm.mlpl`
body in `device("mlx") { ... }` and otherwise changes nothing.
Measurable outcome: identical-shape loss curve, >=5x wall-clock
speedup over the interpreter CPU path on an M-class laptop
(target 10-50x per `docs/using-mlx.md`, but 5x is the go / no-go
gate). Extend `mlpl-bench` with an MLX row on the existing
reshape+reduce workload and the Tiny LM training-step workload.
Document the numbers in `docs/benchmarks.md`.

## Phase 5 -- Docs + release (2 steps)

### Step 009 -- tutorial lesson + using-mlx.md retrospective
New web REPL tutorial lesson **"Running on MLX"** that walks
from a `device("cpu") { randn(7, [1024, 64]) }` baseline to the
same expression wrapped in `device("mlx") { ... }`, then shows
a tiny training loop swap. Update `docs/using-mlx.md` from
"design sketch" to "reference" and strip the `> Status: planned`
disclaimer; add a retrospective section summarizing what
actually shipped vs the sketch. Update `docs/status.md`
one-liner and `docs/saga.md` Saga 14 entry. Rebuild `pages/`
via `scripts/build-pages.sh` and commit both source and built
artifact in the same commit.

### Step 010 -- release v0.11.0
Bump workspace version to `0.11.0`. Release commit summarizes
`mlpl-mlx` runtime, `device("...") { }` scoped form,
`to_device(...)` movement helpers, autograd + optimizers on
MLX, and the Tiny LM MLX variant with its measured speedup.
Tag `v0.11.0`. Push commit and tag. Verify the pages workflow
deploys the updated tutorial list.

## Dependencies and risk

- Step 001 depends on `mlx-rs` building cleanly on the dev
  laptop; pin a known-good version in `Cargo.toml`. Risk: MLX
  crate ABI changes between versions. Mitigation: pin minor.
- Step 004 introduces `Expr::Device`. Risk: the scoped form
  interacts with `experiment { }` and `train { }` nesting.
  Mitigation: explicit tests for `experiment { device { train
  { } } }` and the swapped nesting.
- Step 006 is the highest-risk step. MLX's lazy graph and
  MLPL's tape are two autograd systems; we only get to pick
  one per expression. Decide during TDD whether to lean on
  `mlx-rs`'s `vjp` / `grad` wholesale or hand-write backward
  per primitive. Either path must produce the same gradients
  (within tolerance) as the CPU tape.
- Step 008 risk: MLX compile overhead on the first call may
  dominate the Tiny LM's wall clock at this size. Mitigation:
  bench harness reports both cold and warm timings; go/no-go
  is warm-path speedup.
- Cross-platform CI risk: every MLX test must be feature- and
  target-gated so `cargo test` on Linux CI stays green. This
  is an invariant across every step, not a single-step task.
