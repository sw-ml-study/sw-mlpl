# Perturbation Research Demos Milestone (Saga 20, v0.12.0)

## Why this exists

Saga 13 trained a Tiny LM on CPU; Saga 14 put the same forward
pass on MLX with interactive latency for many independent
variants. That unlocks a class of research workflow the platform
has not yet expressed as MLPL source: weight-perturbation
specialization, a.k.a. [Neural Thickets / nt-rs](https://github.com/swcraig/nt-rs)
(the Rust research harness for the [RandOpt](https://www.alphaxiv.org/overview/2603.12228.pdfv1)
approach).

The thesis: in a pretrained model the weight-space neighborhood
is dense with task-specialized solutions; sample N perturbations,
score, take top-K, ensemble. MLPL already owns forward/autograd,
named params, `randn`, `top_k`, `device("mlx") { }`, and
`svg(..., "heatmap")`. Four small builtins close the gap:
`clone_model`, `perturb_params`, `argtop_k`, `scatter`.

Saga 20 ships those builtins, a `demos/neural_thicket.mlpl`
headline demo that trains a Tiny LM, sweeps four perturbation
families on MLX, top-K ensembles, and renders a
`[family x seed]` specialization heatmap, plus a tutorial lesson
and `docs/using-perturbation.md` retrospective. Release is
v0.12.0.

See `docs/mlpl-for-neural-thickets.md` for the design sketch and
strawman source.

## Non-goals

- **No real pretrained LLM.** Demo runs on the Saga 13 Tiny LM.
  A Llama-class host would need checkpoint loading (Saga 15+) or
  an Ollama sidecar (Saga 19).
- **No distributed coordination.** nt-rs's coordinator + worker
  shard pattern is Saga 17 (CUDA + LAN).
- **No new optimizer.** Variants are scored on a frozen forward;
  `adam` trains the base only.
- **No new viz primitive.** Existing `svg(..., "heatmap")` and
  `svg(..., "bar")` cover the visualization story.
- **No `early_N_layers` / `late_N_layers` families.** They need
  explicit layer depth in the param name, which we do not encode.
  Ship the four families the nt-rs surface calls out
  (`all_layers`, `attention_only`, `mlp_only`, `embed_and_head`);
  defer depth-aware families to a follow-up.
- **No low-rank perturbation.** `perturb_low_rank` is a sibling
  to `perturb_params`; defer until the family walker exists and
  proves out on the Gaussian case.

## Quality requirements (every step)

Identical to Saga 14:

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
7. MLX-feature tests stay gated behind
   `#[cfg(all(target_os = "macos", target_arch = "aarch64",
   feature = "mlx"))]` (or the equivalent `mlpl-mlx` gate).
   The CPU path is authoritative; MLX paths mirror its
   numerics within documented fp32 tolerance.

## What already exists

- Saga 13 Tiny LM end-to-end (`demos/tiny_lm.mlpl`), including
  `embed`, `chain`, `residual`, `rms_norm`, `causal_attention`,
  `linear`, `relu_layer`, `cross_entropy`, `adam`, `train N { }`.
- Saga 14 MLX backend: `to_device(model, "mlx")`, the
  `device("mlx") { ... }` scoped form, MLX autograd via tape,
  `mlpl-bench` Criterion harness with `tiny_lm_mlx_demo`.
- `randn(seed, shape)` and per-seed reproducibility (Saga 8).
- `top_k(logits, k)` -- masked logits for sampling (Saga 13).
- Model DSL param name conventions: `__attn_Wq_*`,
  `__attn_Wk_*`, `__attn_Wv_*`, `__attn_Wo_*`, `__linear_W_*`,
  `__linear_b_*`, `__embed_E_*`, `__rmsnorm_*` (Sagas 11/13).
- `svg(matrix, "heatmap")` and `svg(vec, "bar")` (Saga 7).
- `reshape`, `shape`, `reduce_mul`, `mean`, `neg`, `zeros`,
  integer indexing on vectors.

## Phase 1 -- Model mutation builtins (3 steps)

### Step 001 -- `clone_model` builtin
Deep-copy a `ModelSpec` tree: walk every layer, allocate fresh
param names (stable rename scheme, e.g. suffix with a fresh
clone id so repeated clones stay distinct), copy the stored
values, return a new `ModelSpec` independent of the original.
Mutating the clone's params through `perturb_params` or `adam`
must not affect the source. Contract lives in
`contracts/eval/clone-model.md` as prose; test lives in
`crates/mlpl-eval/tests/clone_model_tests.rs`.

TDD shape: construct a small `chain(linear(4, 4, 0),
linear(4, 4, 1))` base, clone it, mutate the clone's params (or
train it one step), assert the base's params are unchanged; and
assert `clone_model(m)` has the same forward output as `m` on
a fixed input (identity before perturbation).

### Step 002 -- `perturb_params` + family pattern walker
Walk a `ModelSpec`'s params, filter by family name, and add
`sigma * randn(seed, shape)` to each matching param in place.
Families:

- `all_layers` -- every param.
- `attention_only` -- `__attn_Wq_*`, `__attn_Wk_*`,
  `__attn_Wv_*`, `__attn_Wo_*`.
- `mlp_only` -- `__linear_W_*` / `__linear_b_*` outside the final
  projection head.
- `embed_and_head` -- `__embed_E_*` plus the final `linear`'s
  `W` / `b`.

The "final linear projection head" discrimination is the one
subtle bit -- the walker needs to know which `__linear_*` belongs
to the head vs. the MLP. Use the structural position in the
`ModelSpec` tree (last top-level `linear` child of the outermost
`chain`), not a name pattern; this keeps the test
deterministic.

Unknown family strings raise `EvalError::InvalidArgument` with
the list of accepted families in the message.

TDD shape: clone a base, call `perturb_params(clone,
"attention_only", 0.02, 42)`, assert attention params differ
from base's within `abs(delta) in [0, 6*sigma]`, and MLP /
embed / head params are bit-identical to base's.

### Step 003 -- `argtop_k` + `scatter` builtins
Two small utility builtins that close the ensemble loop:

- `argtop_k(values, k)` -- returns the `k` indices (as a rank-1
  integer array) of the largest entries in `values`. Companion
  to the existing `top_k(logits, k)` (which masks logits);
  different name because the return type differs.
- `scatter(buffer, index, value)` -- returns `buffer` with
  `buffer[index] = value`. Rank-1 buffer, scalar `index`, scalar
  `value`. Pairs with `repeat N { ... }` loops that produce one
  number per iteration.

Both land in the interpreter and -- if they fit the existing
pattern trivially -- in `mlpl-rt` too. If the `mlpl-rt` side is
non-trivial, document the gap and leave a follow-up issue;
Saga 20's demo runs through the interpreter either way.

TDD shape: `argtop_k([0.1, 0.5, 0.2, 0.9], 2)` equals
`[3, 1]`; `scatter(zeros([4]), 2, 7.5)` equals `[0, 0, 7.5,
0]`; `argtop_k` of a tie breaks by lowest index first;
out-of-range index raises `EvalError::IndexOutOfBounds`.

## Phase 2 -- Headline demo (2 steps)

### Step 004 -- `demos/neural_thicket.mlpl` on CPU
First end-to-end runnable version of the demo, on CPU (no
`device("mlx")` block yet). Trains a base Tiny LM (shorter run
than Saga 13 -- the demo is about perturbation, not base
quality), sweeps four families x four seeds = 16 variants,
scores each on held-out validation tokens, builds a
`losses : [16]` vector via `scatter`, picks top-K via
`argtop_k`, reshapes to `[family, seed]` heatmap matrix.

Demo runs clean under `mlpl-repl -f demos/neural_thicket.mlpl`
(remember: stdin piping splits `repeat { }` -- always `-f`).
Commit the demo; `docs/demos-scripts.md` gets a one-line entry.

Integration test in `crates/mlpl-eval/tests/neural_thicket_tests.rs`
runs a tiny cut-down version (smaller V, smaller N, 10 training
steps) and asserts the heatmap shape is `[4, 4]`, losses are
finite, and `argtop_k` picks indices in-range.

### Step 005 -- MLX variant loop + bench parity
Wrap the perturb / score loop in `device("mlx") { ... }` with
a `to_device(base, "mlx")` prologue, mirroring the Saga 14
pattern. Extend `mlpl-bench` with a `neural_thicket_mlx`
Criterion entry so the speedup is recorded next to the Tiny LM
bench. Document the measured numbers (whatever they are,
honestly -- Saga 14 already set the precedent that MLX can be
slower than CPU on small workloads; report what we see).

TDD shape: `crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`
that runs the MLX variant-loop cut-down under the existing
MLX cfg gate and asserts heatmap shape + finite losses match
the CPU path's shape within fp32 tolerance.

## Phase 3 -- Tutorials, docs, release (2 steps)

### Step 006 -- Heatmap tutorial lesson + docs
Add a "Neural Thickets" tutorial lesson to the web REPL
following the Saga 14 "Running on MLX" pattern: a short
narrative, the demo source, and a rendered heatmap. Rebuild
`pages/` via `scripts/build-pages.sh` and commit both the
source changes and the rebuilt `pages/` dir (per CLAUDE.md).

Write `docs/using-perturbation.md` -- user-facing retrospective
covering: what the four builtins do, which families matter and
why, how the demo composes, the measured heatmap, honest numbers
on MLX vs CPU for the variant loop, and what the follow-up work
looks like (depth-aware families, low-rank, real checkpoints).

### Step 007 -- Release v0.12.0
Bump workspace version to 0.12.0, update `CHANGELOG.md` with a
Saga 20 entry (new builtins, new demo, new docs, measured
speedups), update `docs/saga.md` to mark Saga 20 complete, tag
the release commit. Final `/mw-cp` run and push. `agentrail
complete --done` closes the saga.

## Dependency graph

```
001 clone_model
  \-- 002 perturb_params
        \-- 003 argtop_k + scatter
              \-- 004 neural_thicket CPU
                    \-- 005 neural_thicket MLX + bench
                          \-- 006 tutorial + using-perturbation.md
                                \-- 007 release v0.12.0
```

Steps 001-003 can in principle interleave (001 is the only hard
prerequisite of 002), but the sequential order keeps the commit
story clean and each step's contract testable in isolation.
