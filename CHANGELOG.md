# Changelog

All notable changes to MLPL. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
canonical per-saga retrospectives live in `docs/saga.md` and
`docs/milestone-*.md`.

## v0.16.0 -- Saga 19: LLM-as-Tool REST Integration (2026-04-24)

A language-level builtin that POSTs to an Ollama-
compatible `/api/generate` endpoint and returns the
model's completion as a string. The REPL's preview
`:ask` command predated this -- it now delegates to
the same shared HTTP path so the machinery lives in
one place. Headline use case is "LLM as a tool":
pipe a value through a hosted model, store the
reply as an MLPL string, feed it to `tokenize_bytes`
/ `experiment {}` / the rest of the pipeline.
CLI-only -- the browser path lands in Saga 21 via
the `mlpl-serve` reverse proxy.

### Added

- **`llm_call(url, prompt, model) -> string`**.
  Single-POST whole-reply form. URL is normalized
  (trailing slashes stripped; `/api/generate`
  appended unless already present, no
  double-append). 120s timeout matches the `:ask`
  preview. Returns a `Value::Str` scalar that
  composes with every existing string-accepting
  builtin. Module:
  `crates/mlpl-runtime/src/llm_builtins.rs` (3 fns:
  `call_ollama`, `resolve_url`, `parse_response`)
  for the pure HTTP path; eval-side dispatcher at
  `crates/mlpl-eval/src/llm_dispatch.rs` (1 fn)
  evaluates the three string args, calls the
  runtime helper, lifts `RuntimeError` ->
  `EvalError::Unsupported`. Contract:
  `contracts/eval-contract/llm-call.md`.
- **`demos/llm_tool.mlpl`**. CLI-only end-to-end:
  call llama3.2 via `llm_call`, print the reply,
  feed it through `tokenize_bytes` to show
  composition. Header guard explains the demo
  needs a running Ollama at localhost:11434 with
  `llama3.2` pulled; CI / no-network envs should
  skip it. Verified live against a local Ollama
  during step 002.
- **`docs/using-llm-tool.md`**. Retrospective +
  user guide. Sections: status, what this is
  about, signature + contract link, Ollama setup,
  configuration (env vars used by `:ask`,
  explicit-arg pattern for the language-level
  form), composition stories
  (`tokenize_bytes`-after-`llm_call`,
  inside-`experiment {}`, distillation pseudocode
  for a future saga), the `:ask` relationship,
  why no web/WASM support, full deferred
  non-goals list.

### Changed

- **`:ask` REPL command** now delegates the HTTP
  path to `mlpl_runtime::call_ollama` (the same
  path `llm_call` uses) instead of carrying its
  own duplicate `ureq` POST. The system framing,
  workspace `:vars` summary, and `_demo`
  narration all stay -- they get concatenated
  into a single prompt string and POSTed through
  `/api/generate`. **Behavior change:** the
  pre-v0.16 `:ask` used Ollama's `/api/chat` with
  explicit `{role: system}` + `{role: user}`
  messages; the new `:ask` uses `/api/generate`,
  which loses the role distinction but keeps the
  context. A future `llm_chat(history, prompt)`
  variant (listed in step 001's deferred non-
  goals) would restore `/api/chat` semantics
  while still sharing the underlying machinery.
  TODO note in `apps/mlpl-repl/src/ask.rs` flags
  this as the migration path.
- **`apps/mlpl-repl/Cargo.toml`** drops `ureq` and
  `serde_json` direct deps; the REPL was the only
  consumer of those crates and the HTTP path now
  lives in `mlpl-runtime`. Replaced with a single
  `mlpl-runtime` path dep.
- **`docs/using-ollama.md`** status block updated
  to redirect to `docs/using-llm-tool.md` as the
  canonical reference; design-phase notes
  preserved for historical context.
- **`docs/configurations.md`** new
  `llm_call(url, prompt, model)` row in the
  CLI-vs-web matrix (CLI yes; web no with the
  Saga 21 proxy pointer); new footnote [7]
  documents the Saga 19 scope and CORS story.

### Tests

- `crates/mlpl-eval/tests/llm_call_tests.rs` (step
  001): 8 mockito-based integration tests covering
  happy path (model + prompt JSON body match
  asserted, returns the `response` field), URL
  auto-append (bare base + trailing slash),
  full-URL passthrough (no double-append on
  `/api/generate/api/generate`), non-2xx status
  (errors with `llm_call`, status code, and
  body preview), missing `response` field,
  invalid JSON body, wrong arity, non-string
  argument.

### Scope notes

- **Streaming SSE** deferred. Single-POST
  whole-reply form; a `llm_stream(...)` variant
  or callback-style argument can land later.
- **OpenAI-style tool calling** deferred. `tools`
  + `tool_choice` request fields plus the
  matching JSON-schema parser on the MLPL side.
  Waits on a concrete use case.
- **Multi-turn chat threading** deferred. The
  builtin is single-turn; `llm_chat(history,
  prompt)` would be the variant.
- **Request batching** deferred. One call per
  invocation.
- **In-source auth secrets** deferred and
  intentionally out-of-scope. No bearer-token /
  API-key argument; secrets must come from
  process env vars or the CLI server's
  server-side allow-list, never from MLPL source.
- **Teacher-model distillation pipeline**
  deferred. Uses `llm_call` to generate soft
  labels; separate saga once dtype machinery for
  fp16/bf16 is in place.
- **Web / WASM support** deferred. Browser CORS
  blocks localhost Ollama; Saga 21's
  `mlpl-serve` reverse proxy is the path.

## v0.15.0 -- Saga 22: Feasibility Checking + Resource Estimation (2026-04-24)

Four new builtins that let the user sanity-check a
planned training run BEFORE committing disk, RAM, and
hours of wall-clock to it. Pure math over a
`ModelSpec` (or a hardcoded HF-scale dimension table)
plus train-loop parameters; no weights required for
what-if queries. Targets ~2x accuracy as an honest
lower bound -- activation memory is a safety-factor
heuristic, the FLOPS model ignores softmax / layer
norm / elementwise costs, and wall-clock reads
throughput from a one-shot calibration benchmark.

### Added

- **`estimate_train(model, steps, batch_size, seq_len
  [, dtype_bytes]) -> [5]`**. Rank-1 f64 array
  `[params, vram_bytes, disk_bytes, flops,
  wall_seconds]` -- labels pinned in the contract. VRAM
  sums forward weights (all params) + gradient +
  Adam m/v moments (trainable only; frozen LoRA base
  is zero on the grad/adam legs) + an activation
  estimate `batch * seq * hidden * depth *
  dtype_bytes * 4`. Disk is one full checkpoint.
  FLOPS model covers Linear / LinearLora / Attention
  / Embedding. Wall-clock = flops /
  `mlpl_device_throughput_gflops` (defaults 50
  GFLOPS as a CPU laptop lower bound). Default
  `dtype_bytes = 8` (f64); pass 4 for f32 what-ifs,
  2 for f16/bf16. Contract:
  `contracts/eval-contract/estimate.md`. Module:
  `crates/mlpl-runtime/src/estimate_builtins.rs` (6
  fns, within the 7-fn budget).
- **`calibrate_device() -> gflops`**. Zero-arg
  benchmark: 10 iterations of a 1024x1024 matmul
  through `device::dispatched_call`, wall-clock
  measured (first iter discarded as warmup).
  Observed GFLOPS is written into
  `env.set_string("mlpl_device_throughput_gflops",
  ...)` so subsequent `estimate_train` calls read
  honest numbers. Device-aware -- under
  `device("mlx") { ... }` writes
  `mlpl_device_throughput_gflops_mlx` instead; the
  estimator's lookup is device-aware in the same
  way. Contract:
  `contracts/eval-contract/calibrate-device.md`.
- **`estimate_hypothetical(name, steps, batch_size,
  seq_len [, dtype_bytes]) -> [5]`** (design
  deviation from the original plan, which proposed
  `hypothetical_model(name) -> ModelSpec`).
  `estimate_train` reads parameter shapes from
  `env`, and materializing zero arrays for a
  SmolLM-1.7B spec would cost ~14 GB just to ask
  the question -- hostile for a what-if query.
  Shipped as a direct `[5]`-return builtin that
  consults the same hardcoded SmolLM / Llama /
  Qwen dimension table without populating any
  env entries; output shape is identical to
  `estimate_train` so `feasible(...)` composes
  unchanged. Supported names: `smollm-135m`,
  `smollm-360m`, `smollm-1.7b`, `llama-3.2-1b`,
  `qwen-2.5-0.5b`. Contract:
  `contracts/eval-contract/estimate-hypothetical.md`.
- **`feasible(estimate_result, budget) -> 0/1`**.
  Guard-pattern builtin. `budget` is a rank-1 `[3]`
  `[vram_bytes, disk_bytes, wall_seconds]`; zeros
  skip that dimension. Returns scalar 1.0 if every
  non-zero budget is satisfied, 0.0 otherwise.
  Pairs with `if feasible(est, [4e9, 1e10, 600])
  { train ... }` to abort doomed runs before
  allocating. Contract:
  `contracts/eval-contract/feasible.md`. Module
  (with `calibrate_device` and
  `estimate_hypothetical`):
  `crates/mlpl-eval/src/model_feasibility.rs` (7
  fns at the budget limit).

### Changed

- `demos/feasibility.mlpl` (new, CLI-only):
  estimates a tiny mlpl-toy model, calibrates the
  device, re-estimates (wall drops), compares a
  SmolLM-135M full fine-tune against a LoRA
  fine-tune, gates a mock train call on
  `feasible(...)`.
- `docs/using-feasibility.md` (new): what the
  estimator computes and what it does NOT (the ~2x
  accuracy target, the activation_factor = 4
  heuristic), signature reference for all four
  builtins, the hardcoded HF-scale dimension
  table, the LoRA-on-SmolLM worked example, the
  guard pattern, deferred non-goals.
- `docs/configurations.md`: new rows for
  `estimate_train`, `estimate_hypothetical`, and
  `feasible` (work in web; pure math, no device
  calls) and `calibrate_device` (CLI-only --
  browser timers are too noisy and the WASM
  device has no GPU path to measure).

### Tests

- `crates/mlpl-eval/tests/estimate_tests.rs` (step
  001): 11 tests -- tiny linear exact math (params,
  VRAM legs, FLOPS), two-linear chain additivity,
  LoRA (base frozen, adapters trainable; grad/adam
  count only adapters; disk counts all),
  Embedding params + FLOPS, Attention bumps
  activation bytes vs no-attention chain, 5-arg
  `dtype_bytes = 4` halves VRAM, 4 error paths
  (non-model, negative scalars, non-scalar args,
  no-params model).
- `crates/mlpl-eval/tests/feasibility_tests.rs`
  (step 002): 12 tests (11 fast + 1
  `#[ignore]`-ed slow benchmark for the 1024x1024
  default-size `calibrate_device` run which takes
  minutes on CPU). Covers `calibrate_device`
  (positive GFLOPS, env key cached),
  `estimate_hypothetical` (SmolLM-135M param
  count in the 100-200M range, scales across
  sizes, LoRA drops VRAM while disk is unchanged,
  unknown name / wrong arity errors), and
  `feasible` (passes / fails per-dimension, zero
  skips, wrong shapes error, composes with
  `estimate_train` via the `[5]` / `[3]` flow).

### Scope notes

- HuggingFace Hub download + safetensors loading
  are deferred; `estimate_hypothetical` talks
  about these models structurally WITHOUT
  requiring any weights on disk.
- f16 / bf16 tensor support is deferred -- MLPL
  runs f64 today. The `dtype_bytes` argument is a
  what-if knob for the estimator only.
- Activation memory is a 4x safety multiplier;
  exact numbers need a real profiler.
- Distributed / multi-GPU estimation is Saga 17
  territory (CUDA backend + distributed).
- Auto-recovery / batch-shrinking is a non-goal;
  the estimator reports, the user decides.

## v0.14.1 -- Saga 16.5: Embedding-viz Polish (2026-04-24)

Two convenience builtins that close the loose ends Saga
16 explicitly flagged: a one-line `pca(X, k)` and a
source-level way to pull an embed layer's weights back
out of a trained chain. Demo, docs, and tutorial
updated to use both; no breaking changes.

### Added

- **`pca(X, k) -> Y`**. Top-k principal component
  analysis via power iteration + Gram-Schmidt + deflation
  (50 iterations per component; GS inside the loop
  guarantees orthogonality even when later eigenvalues
  are numerical noise, which matters for `k = D` on rank-
  deficient inputs). Returns the centered-and-projected
  `[N, k]` data; the components themselves are not
  returned (run the Saga 8 composition pattern directly
  if you need them). Contract:
  `contracts/eval-contract/pca.md`. Module:
  `crates/mlpl-runtime/src/pca_builtin.rs` (6 fns,
  within the 7-fn budget).
- **`embed_table(model) -> table`**. Depth-first left-
  to-right walk of a `ModelSpec` tree; returns the first
  Embedding layer's `[vocab, d_model]` lookup table
  cloned from `env`. First-match semantics (documented;
  multi-embedding selectors are deferred). Works inside
  `Chain`, `Residual`, or nested `Chain(Chain(...))`;
  errors cleanly if no Embedding is present. Contract:
  `contracts/eval-contract/embed-table.md`. Module:
  `crates/mlpl-eval/src/model_embed_table.rs` (2 fns).

### Changed

- `demos/embedding_viz.mlpl` now uses `pca(table, 3)`
  for the 3-D projection instead of the column-selector
  matmul. Same cluster structure, principled projection.
- `docs/using-embeddings.md`: PCA section rewritten from
  "composition-only" to "shipped as a builtin in
  v0.14.1" (keeps the power-iteration recipe as
  pedagogical reference); new "Extracting embed-layer
  weights" section; "Training inside a chain" workaround
  story updated to the shipped `embed_table` flow.
- `apps/mlpl-web/src/lessons_advanced.rs` "Embedding
  exploration" lesson adds `pca(X, 2)` and
  `embed_table(emb)` examples alongside the existing
  `pairwise_sqdist` / `knn` / `tsne` / `svg(...,
  scatter3d)` walkthrough. Intro and try_it updated.

### Tests

- `crates/mlpl-eval/tests/pca_tests.rs`: 9 tests --
  anisotropic 2-D capture (>80% variance retained with
  k=1), shape preservation for `1 <= k <= D`, k=D
  variance preservation within 1e-4, determinism, 5
  error paths (rank != 2, k=0, k>D, non-finite, wrong
  arity).
- `crates/mlpl-eval/tests/embed_table_tests.rs`: 8 tests
  -- standalone, chain-at-position-0, residual, nested
  chain, reflects trained weights (diff > 1e-6 after 5
  adam steps), no-embedding error, wrong arity, non-
  model argument.

### Scope notes

- UMAP, interactive 3-D scatter, MLX dispatch for
  `tsne`, and approximate / Barnes-Hut `tsne` remain
  deferred (same reasoning as v0.14.0).
- Multi-embedding path-selector variant of
  `embed_table` (e.g., `embed_table(m,
  "encoder.embed")`) is deferred -- not a shipped
  pattern today.

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
