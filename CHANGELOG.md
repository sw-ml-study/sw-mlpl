# Changelog

All notable changes to MLPL. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
canonical per-saga retrospectives live in `docs/saga.md` and
`docs/milestone-*.md`.

## v0.14.0 -- Saga 16: Embedding Visualization (2026-04-23)

Embedding-inspection surface for the Model DSL. Three new
builtins + one new SVG type close the "train a model,
inspect what it learned" loop, and a new decomposition-
patterns doc formalizes the design rules that kept Sagas
15 and 16 under the sw-checklist budgets.

### Added

- **`pairwise_sqdist(X) -> D`** (step 001). Rank-2
  `[N, D]` -> rank-2 `[N, N]` squared Euclidean
  distances, `D[i, j] = sum_k (X[i, k] - X[j, k])^2`.
  Symmetric, zero diagonal, empty-input safe. Contract:
  `contracts/eval-contract/pairwise-sqdist.md`.
- **`knn(X, k) -> idx`** (step 001). Each row's `k`
  nearest non-self neighbors sorted by ascending
  distance, ties broken by lower original index. Row
  `i` never appears in `idx[i]`. Contract:
  `contracts/eval-contract/knn.md`.
- **`tsne(X, perplexity, iters, seed) -> Y`** (step
  002). Classic van der Maaten t-SNE: per-row
  perplexity-calibrated binary search for beta,
  symmetrized joint probabilities, Student-t low-dim
  affinities, KL-loss gradient descent with
  early exaggeration. Deterministic under identical
  seeds; rotational / reflection symmetry in absolute
  coordinates. Four sibling modules
  (`tsne_builtin` / `tsne_validate` / `tsne_affinities`
  / `tsne_gradient`) following the decomposition
  patterns. Contract:
  `contracts/eval-contract/tsne.md`.
- **`svg(pts, "scatter3d")`** (step 003). Static 3-D
  scatter via orthographic projection at fixed
  azimuth=30 / elevation=20. Accepts `[N, 3]` or
  `[N, 4]` (points + integer cluster id); renders
  labeled axis gizmos + one `<circle>` per point +
  optional sorted legend. Snapshot-deterministic.
  Contract: `contracts/viz-contract/scatter3d.md`.
- **`demos/embedding_viz.mlpl`** (step 004, CPU, any
  host). Trains a standalone `embed(12, 8, 0)` toward
  a structured 3-cluster target via MSE for 50 adam
  steps; extracts the learned table via
  `apply(emb, iota(V))`; runs it through t-SNE (2-D),
  a column-selector matmul (3-D), and `knn` (neighbor
  reporting). Final neighbor output confirms learned
  embeddings recover the target structure (every
  token's nearest neighbors are in its own cluster).
- **"Embedding exploration" web REPL tutorial lesson**
  (step 005). Tiny 6-point 3-D fixture renders
  instantly in WASM; walks all three builtins + 3-D
  scatter.
- **`docs/using-embeddings.md`** (step 005). User-
  facing retrospective: the three builtins, the
  scatter3d viz type, the PCA composition pattern,
  demo walkthrough, parity testing inventory (5 test
  files, 35 tests), and the deferred follow-ups.
- **`docs/sw-checklist-patterns.md`** (step 002, shipped
  mid-saga). Decomposition patterns catalog: why the
  budgets exist, pre-implementation checklist,
  worked examples (struct returns, struct args,
  validate-then-work, orchestrator + helpers, per-
  variant helpers, phase helpers, extract-to-module,
  chain-of-responsibility, bridge, split-tests-from-
  work), anti-patterns, and the "new commits don't
  add new FAILs, edits to over-budget fns trigger
  extractions" rules. Companion
  `feedback_sw_checklist_patterns.md` memory for
  cross-session persistence.

### Refactored

- Extracted the three "advanced" web REPL lessons
  (Neural Thickets from Saga 20, LoRA Fine-Tuning
  from Saga 15, new Embedding exploration from Saga
  16) into a sibling `apps/mlpl-web/src/lessons_advanced.rs`
  module. `lessons.rs` shrinks from 500 (at budget)
  to 467; `lessons_advanced.rs` sits at 62. Both
  under the 500-LOC budget; new lessons added in
  either module cleanly.

### Scope notes

- **PCA stays a composition pattern, not a builtin.**
  The power-iteration + deflation recipe is already a
  Saga 8 tutorial lesson; a `pca(X, k)` wrapper would
  hide mechanics that deserve being visible. Becomes
  trivial to add if an ergonomic use case emerges.
- **t-SNE is CPU-only.** Its inner loop does not
  vectorize cleanly through MLX's kernel model at
  embedding-table scale. `pairwise_sqdist` and `knn`
  are also CPU-only primitives. Deferred.
- **`embed_table(model)` is not shipped.** The demo
  uses a standalone `embed` layer + MSE-to-target to
  sidestep the chain-child-can't-be-a-bound-ident
  language quirk; `embed_table` would be the clean
  fix (one-fn addition, not done this saga).

### Not shipped (deferred follow-ups)

- UMAP (sibling nonlinear reducer; overlaps with
  t-SNE for marginal user value).
- `pca(X, k)` builtin.
- RAG pipeline (pending Saga 19 LLM-as-tool REST
  integration).
- Interactive 3-D scatter with rotation/zoom.
- MLX dispatch for `tsne`.
- `embed_table(model)` builtin.
- Barnes-Hut approximate t-SNE (exact O(N^2) is fine
  at embedding-table scale).

See `docs/using-embeddings.md` "Not shipped" section.

## v0.13.0 -- Saga 15: LoRA Fine-Tuning (2026-04-23)

Parameter-efficient fine-tuning lands in MLPL source. Three
new builtins plus a new `ModelSpec` variant compose into a
PyTorch-`peft`-style "frozen base, trainable low-rank
adapters" workflow: take a trained base, wrap it with
`lora(base, rank, alpha, seed)`, train only the adapters.

### Added

- **`freeze(m) -> scalar 0`** (step 001). Marks every name
  in `m.params()` as frozen in `env.frozen_params`. `adam`
  and `momentum_sgd` skip frozen names at the optimizer-
  update stage; gradient computation is unchanged, so the
  chain rule still flows through for downstream ops.
  Contract: `contracts/eval-contract/freeze.md`.
- **`unfreeze(m) -> scalar 0`** (step 001). Inverse of
  `freeze`. Removes every `m.params()` name from
  `env.frozen_params`. Idempotent.
- **`lora(m, rank, alpha, seed) -> Model`** (step 002).
  Clones `m`'s spec tree, replaces every `Linear` node
  with a `LinearLora` that owns the cloned base `W`, `b`
  plus two fresh low-rank adapter matrices `A [in, rank]`
  (init `randn * 1/sqrt(in)`) and `B [rank, out]` (init
  zeros), and auto-freezes every non-adapter param in the
  returned student. The zero-init on B gives the LoRA
  "forward identity before training" property
  (`apply(lora_m, X) == apply(m, X)` elementwise). Contract:
  `contracts/eval-contract/lora.md`.
- **`ModelSpec::LinearLora`** (step 002). New variant on
  the `ModelSpec` enum; `in_dim, out_dim, rank, alpha`
  cached for the forward formula. `params()` returns
  `[w, b, a, b_adapter]`.
- **Forward + autograd for `LinearLora`** (step 003).
  `apply_model` and `apply_model_tape` compute
  `y = X @ W + (alpha / rank) * X @ A @ B + b`. Threaded
  through the existing matmul / scalar-mul / add dispatch,
  so MLX (Saga 14) picks it up for free; no new tape ops
  were needed.
- **`demos/lora_finetune.mlpl`** (step 004, CPU, any host).
  Pre-train a Saga 13 Tiny LM on the Shakespeare snippet
  (100 Adam steps), wrap with rank-8 LoRA, fine-tune the
  adapters on a synthetic Q/A instruction corpus (50
  steps). Final fine-tune cross-entropy ~2.18; base
  bit-identical throughout.
- **`demos/lora_finetune_mlx.mlpl`** (step 005, Apple
  Silicon CLI). Mirror with the fine-tune loop wrapped in
  `device("mlx") { ... }`. Base pre-train stays on CPU.
- **Criterion bench group `lora_finetune_step`** (step 005)
  in `crates/mlpl-bench/benches/mlx_vs_cpu.rs`.
- **Web REPL tutorial lesson "LoRA Fine-Tuning"** (step 006,
  `apps/mlpl-web/src/lessons.rs`). Tiny interactive variant
  (V=8, d=4, rank=2) so a 10-step fine-tune renders in
  WASM.
- **`docs/using-lora.md`** (step 006). User-facing
  retrospective covering the three-builtin surface,
  initialization conventions, auto-freeze semantics
  (including the step 004 amendment covering embed +
  attention, not just Linear), device propagation, demo
  walkthroughs, measured MLX numbers, parity testing, and
  the deferred follow-up list.

### Measured

On an M-class laptop (`cargo bench -p mlpl-bench --features
mlx --bench mlx_vs_cpu -- lora_finetune_step`):

| Path | Cold | Warm |
|---|---:|---:|
| CPU | 208 us | 164 us |
| MLX | 1.45 ms | 1.11 ms |

MLX is **0.15x** of CPU on this workload -- a step down
from Saga 14's `tiny_lm_train_step` (0.26x) and Saga 20's
`neural_thicket_variant_loop` (0.25x) at the same Tiny LM
scale. Cause: LoRA doubles the matmul count per linear
(`X @ W` AND `X @ A @ B`) and the rank-2 adapter matmuls
are too small to amortize MLX's per-op kernel-launch
overhead. At d=512 the ratio would flip. See
`docs/benchmarks.md` for the full analysis (bottleneck
categories unchanged from Saga 14).

Correctness: CPU-vs-MLX fine-tune losses AND every student
param agree elementwise within fp32 tolerance (1e-3);
frozen base bit-identical on both paths (confirming the
optimizer's frozen filter is backend-independent). See
`crates/mlpl-eval/tests/lora_mlx_demo_tests.rs`.

### Refactored

- Step 002's auto-freeze originally only covered the MLP
  `Linear`'s cloned W, b. Step 004 amended `eval_lora` to
  freeze every non-adapter param in the student tree --
  embed tables, attention projections, and MLP base
  linears all auto-freeze now, matching the LoRA library
  convention. One post-rewrite pass, one source of truth;
  the per-Linear `mark_frozen` in `wrap_linear` was
  removed.

### Scope notes

- LoRA language surface is pure Rust and lands identically
  in the CLI REPL and the browser WASM. The MLX
  acceleration path is Apple Silicon + `--features mlx`
  (CLI only); `docs/configurations.md` has the CLI-vs-web
  matrix with Saga 21 as the eventual path to
  MLX-accelerated LoRA from the browser via
  `mlpl-serve`.

### Not shipped (deferred follow-ups)

- QLoRA / 4-bit quantization. Needs per-tensor
  scale/zero-point handling and its own parity harness;
  deferred to a future saga.
- Selective layer attachment (e.g. LoRA only on attention
  projections). Saga 15 ships the uniform "every Linear
  gets an adapter" variant; a `lora(m, ..., layers:
  "attention_only")` variant composes naturally with
  Saga 20's family walker.
- Adapter merging (`merge_lora(m)`).
- Multi-adapter composition / adapter routing.
- Real pretrained LLM checkpoints (needs Saga 15+
  checkpoint format or Saga 19's LLM sidecar).
- Nested `lora()` (currently an explicit error).

See `docs/using-lora.md` "Not shipped" section and
`docs/saga.md` Saga 15 entry.

## v0.12.0 -- Saga 20: Neural Thickets (2026-04-22)

Weight-perturbation research workflow lands in MLPL source.
Four new builtins plus a headline demo compose into a
RandOpt / Neural-Thickets-style specialization search with a
first-class `[family x seed]` heatmap.

### Added

- **`clone_model(m) -> Model`** (step 001). Deep-copies a
  `ModelSpec` tree and allocates a disjoint set of fresh
  param names; device tags (`mlx` vs `cpu`) propagate from
  source to clone. Contract:
  `contracts/eval-contract/clone-model.md`.
- **`perturb_params(m, family, sigma, seed)`** (step 002).
  Adds `sigma * randn(seed + i, shape)` in place to each
  parameter of the named family. Families:
  `all_layers`, `attention_only`, `mlp_only`,
  `embed_and_head`. The "final projection head" is detected
  structurally (last top-level `Linear` child of the
  outermost `Chain`), not by name pattern, so `mlp_only` /
  `embed_and_head` can tell MLP linears apart from the
  vocab projection. Contract:
  `contracts/eval-contract/perturb-params.md`.
- **`argtop_k(values, k)`** (step 003). Index-returning
  companion to the existing `top_k(logits, k)`. Returns the
  k indices of the largest entries in a rank-1 vector,
  sorted by descending value with ties broken by lower
  original index first (stable sort). Contract:
  `contracts/eval-contract/argtop-k.md`.
- **`scatter(buffer, index, value)`** (step 003). Functional
  rank-1 scalar write; returns a copy of `buffer` with the
  single entry at `index` replaced by `value`. Contract:
  `contracts/eval-contract/scatter.md`.
- **`demos/neural_thicket.mlpl`** (step 004, CPU). Full
  end-to-end: train a Saga 13 Tiny LM (V=280, d=32) for 100
  Adam steps, sweep 4 families x 4 seeds = 16 variants,
  score each on a held-out string, heatmap the
  `[family x seed]` specialization, argtop_k the top-K
  specialists, average logits across all 16 for an
  ensemble cross-entropy.
- **`demos/neural_thicket_mlx.mlpl`** (step 005). Same
  source with the variant loop + ensemble under
  `device("mlx") { ... }`. Base training stays on CPU.
- **Criterion bench group `neural_thicket_variant_loop`**
  (step 005) in `crates/mlpl-bench/benches/mlx_vs_cpu.rs`.
- **Web REPL tutorial lesson "Neural Thickets"** (step 006,
  `apps/mlpl-web/src/lessons.rs`). Interactive 4x4 sweep at
  V=8/d=4 so the full heatmap renders in the browser.
- **`docs/using-perturbation.md`** (step 006). User-facing
  retrospective covering the four builtins, the four
  families, the heatmap narrative, honest MLX-vs-CPU
  numbers, parity testing, and the deferred follow-up
  surface.

### Measured

On an M-class laptop (`cargo bench -p mlpl-bench --features mlx
--bench mlx_vs_cpu -- neural_thicket`):

| Path | Cold | Warm |
|---|---:|---:|
| CPU | 838 us | 767 us |
| MLX | 3.12 ms | 3.01 ms |

MLX is **0.25x** of CPU on this workload, essentially
identical to the Saga 14 Tiny-LM training-step ratio
(0.26x). The Neural Thickets workload is inference-only (no
autograd, no tape rematerialization), yet the ratio is
unchanged -- evidence that the bottleneck at Tiny-LM inner
dimensions is per-op kernel launch + f32 round-trip, not
autograd-specific cost. See `docs/benchmarks.md` for the
four-way bottleneck analysis.

Correctness: per-variant losses + ensemble logits agree
with the CPU path elementwise within fp32 tolerance (1e-3),
pinned by `crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`.

### Refactored

- Extracted `Lesson` struct + `LESSONS` const from
  `apps/mlpl-web/src/tutorial.rs` into a new
  `apps/mlpl-web/src/lessons.rs` module. Keeps both files
  under the sw-checklist 500-LOC budget after adding the
  Neural Thickets lesson.

### Not shipped (deferred follow-ups)

- Depth-aware families (`early_N_layers`, `late_N_layers`).
  Need explicit layer indices in parameter names, which the
  Model DSL does not encode today.
- Low-rank perturbation (`perturb_low_rank`). Sibling
  builtin to `perturb_params`; deferred until a concrete
  use case lands.
- Strict top-K ensembling. CPU demo picks `best_idx` via
  `argtop_k` but averages all 16 variants rather than
  rebuilding only the best four. Rebuilding requires
  selecting the family string per flat index, which needs
  either string-array indexing or an index-to-family lookup.
- Real pretrained checkpoints. The demo runs on a
  from-scratch Tiny LM; Llama-class bases need Saga 15+
  (checkpoint format) or Saga 19 (LLM sidecar).
- Per-iteration counter in `repeat`. `train N { }` binds
  `step`; `repeat N { }` does not. The demo works around
  this with `for i in [0, 1, 2, 3] { ... }` per family.

See `docs/using-perturbation.md` "Not shipped" and
`docs/saga.md` Saga 20 entry.

## v0.11.0 -- Saga 14: MLX backend (2026-04-21)

First accelerator backend for MLPL. A program that trains on
CPU runs on Apple Silicon via MLX without source changes.
Ten steps across five phases; correctness complete,
throughput honest. See `docs/saga.md` Saga 14 entry and
`docs/using-mlx.md` for details.

## Prior releases

See `docs/saga.md` and `docs/status.md` for the full
per-saga history back to v0.1. CHANGELOG entries below this
point are maintained as a forward-looking record from the
v0.12.0 cut onward; earlier per-release notes live in the
release commits (search
`git log --grep "^release("`).
