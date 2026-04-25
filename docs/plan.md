# MLPL Multi-Saga Plan

This plan sequences the remaining work from `docs/research2.txt`
(the small-LLM / training-platform vision) against the current
state of the repo. Completed sagas are listed in `docs/saga.md`;
through v0.4 we have the language core, REPL, SVG viz, ML
built-ins, and six end-to-end demos, but every demo still carries
its own hand-written gradients. Everything downstream of autograd
depends on removing that limitation.

## Guiding principles

- Each saga delivers a visible user-facing win (a demo or a
  workflow), not just plumbing.
- Preserve the "thinking space" loop: train -> visualize ->
  adjust. Visualization is core syntax, not an afterthought.
- Backends (MLX, CUDA) come only after the CPU path is correct,
  tested, and expressive enough to be worth accelerating.
- Keep the cellular crate layout; new capabilities land as new
  crates, not as bulges in existing ones.
- TDD + quality gates per CLAUDE.md on every step.

## Completed sagas (context)

- **Saga -1** Repo scaffolding
- **Saga 0** Foundation and contracts
- **Saga 1** Dense tensor substrate v1 (Shape, DenseArray, reshape, transpose, indexing)
- **Saga 2** Parser and evaluator foundation
- **Saga 3** CLI and REPL v1
- **Saga 4** Structured trace v1 (JSON export)
- **Saga 5** Visual web viewer v1 -- DEFERRED
- **Saga 6** ML foundations (v0.2): sigmoid, tanh, pow, comparisons,
  axis reductions, mean, zeros/ones/fill, logistic regression demo
- **Saga 7** SVG visualization v1 (v0.3): `mlpl-viz`, `svg()`,
  scatter/line/bar/heatmap/decision_boundary, hist, scatter_labeled,
  loss_curve, confusion_matrix, boundary_2d, browser REPL inline SVG
- **Saga 8** ML demos (v0.4): random/randn/argmax/blobs/softmax/one_hot;
  k-means, PCA, softmax classifier, tiny MLP, attention pattern demos;
  tutorial lessons for each
- **Saga 9** Autograd v1 (v0.5): `mlpl-autograd` reverse-mode tape;
  `param[...]` / `tensor[...]` constructors; `grad(expr, wrt)` built-in;
  tiny MLP and softmax classifier demos ported off hand-written backprop;
  "Automatic differentiation" tutorial lesson
- **Saga 10** Optimizers + training loop (v0.6): `momentum_sgd` and
  `adam` built-ins with per-param state on the environment;
  `cosine_schedule` / `linear_warmup` scalar helpers; `moons` and
  `circles` synthetic datasets; new `train N { body }` construct
  with implicit `step` binding and `last_losses` capture;
  `moons_mlp` and `circles_mlp` demos; "Optimizers and Schedules"
  tutorial lesson
- **Saga 11.5** Named axes and shape introspection (v0.7.5):
  `LabeledShape` on `mlpl-core`; `label` / `relabel` /
  `reshape_labeled` builtins; `x : [batch, dim] = ...` annotation
  syntax; label propagation through elementwise, matmul,
  reduce/argmax, and `map()`; axis-name arg for
  `reduce_add`/`reduce_mul`/`argmax`/`softmax`; structured
  `EvalError::ShapeMismatch` with op-aware Display; label-aware
  `:vars`/`:describe` and trace JSON; "Named Axes" tutorial
  lesson and labeled Model Composition example
- **Compile-to-Rust saga** (v0.8.0): `mlpl-rt` runtime target,
  `mlpl-lower-rs` AST -> `TokenStream` codegen (scalar, array,
  fncall, variable, labels, matmul with static contraction
  check), `mlpl-macro` proc-macro, `mlpl` facade crate with
  hidden `__rt` re-export, `mlpl-build` CLI that compiles a
  `.mlpl` file to a native binary (and cross-compiles to
  wasm32-unknown-unknown), `mlpl-parity-tests` harness proving
  bit-for-bit agreement on 9 programs. Measured 9.05x speedup
  on a 100x100 reshape+reduce workload. See
  `docs/milestone-compile-to-rust.md`.
- **Saga 12** Tokenizers, datasets, experiment tracking
  (v0.9.0): `load`/`load_preloaded` with `--data-dir` sandbox;
  `shuffle`/`batch`/`batch_mask`/`split`/`val_split` dataset
  prep; `for row in ds { body }` streaming iteration;
  byte-level tokenizer (`tokenize_bytes`/`decode_bytes`);
  `Value::Tokenizer` + `TokenizerSpec::{ByteLevel,BpeMerges}`
  with `train_bpe`/`apply_tokenizer`/`decode`; `experiment
  "name" { body }` with `_metric` capture and `ExperimentRecord`
  logged to memory + `--exp-dir/<name>/<ts>/run.json`;
  `:experiments` + `compare(a, b)` registry; UTF-8 string-
  literal fix in the lexer; string-valued variable bindings.
  Three new tutorial lessons. See `docs/milestone-tokenizers.md`.
- **Saga 11** Model DSL (v0.7): `Value::Model` runtime value;
  `linear(in, out, seed)` atomic layer; parameter-free
  `tanh_layer` / `relu_layer` / `softmax_layer` activations;
  `chain(...)` sequential composition; `residual(block)` skip
  connections; `rms_norm(dim)` normalization; `attention(
  d_model, heads, seed)` multi-head self-attention (tape-lowered
  for `heads=1`); `params(model)` walker and optimizer
  integration so `adam(loss, mdl, ...)` walks the parameter
  tree; differentiable `apply(mdl, X)`; `moons_mlp` and
  `tiny_mlp` demos ported to one-line `chain(...)` form; new
  `transformer_block.mlpl` demo (loss 143.87 -> 1.02 over 100
  Adam steps); `:vars` / `:models` / `:fns` / `:wsid` /
  `:describe` REPL introspection commands; "Model Composition"
  tutorial lesson. See `docs/milestone-modeldsl.md`.

## Future saga sequence

### Saga 11.5 -- Named axes and shape introspection (COMPLETE, v0.7.5)
Shipped: `LabeledShape` type on `mlpl-core`; `label(x, [...])` /
`relabel(x, [...])` / `reshape_labeled(x, dims, labels)` builtins;
`x : [batch, time, dim] = ...` annotation syntax; label propagation
through elementwise (with `merge_labels`), matmul (contraction
axis validated, outer dims threaded), reduce/argmax (reduced
axis's label dropped), and `map()` (preserves through math
builtins); axis-name arg for `reduce_add` / `reduce_mul` /
`argmax` / `softmax`; structured `EvalError::ShapeMismatch { op,
expected, actual }` with op-aware Display; labels in `:vars` /
`:describe` (via `LabeledShape` Display) and in the trace JSON
export (serde-skip-when-None); "Named Axes" tutorial lesson and
labeled `apply(mdl, X)` in the Model Composition lesson.
Deferred: per-layer input/output label pinning on `:describe
<model>` (needs `Environment` signature tracking on first
`apply` -- a natural follow-up).

### Compile-to-Rust saga (COMPLETE, v0.8.0)
Shipped: `mlpl-rt` runtime target, `mlpl-lower-rs` (path-
configurable AST -> Rust TokenStream codegen), `mlpl-macro`
proc-macro (`mlpl! { ... }` emits `compile_error!` on lower
failure), `mlpl` facade crate, `mlpl-build` CLI
(`mlpl-build foo.mlpl -o foo [--target <triple>]`),
`mlpl-parity-tests` harness. Static matmul contraction check
at lower time; runtime label checks still catch anything the
static pass cannot resolve. 9.05x measured speedup on a 100x100
reshape+reduce workload. Deferred: `TensorCtor`, `Repeat`,
`Train`, autograd, optimizers, Model DSL -- they need tape-
state or loop lowering and sit in a follow-up.

### Saga 12 -- Tokenizers, datasets, and experiment tracking (COMPLETE, v0.9.0)
Shipped: `load`/`load_preloaded` with terminal-REPL
`--data-dir` sandbox and compiled-in web corpus registry;
dataset prep (`shuffle`, `batch`, `batch_mask`, `split`,
`val_split`); `for row in ds { body }` streaming iteration
with `last_rows` capture; byte-level BPE
(`tokenize_bytes`/`decode_bytes` primitives;
`Value::Tokenizer` + `TokenizerSpec::{ByteLevel,BpeMerges}`;
`train_bpe`/`apply_tokenizer`/`decode` with byte-lossless
round-trip); experiment tracking
(`experiment "name" { body }`, `_metric`-capture,
`ExperimentRecord` to `env.experiment_log` + optional
on-disk `<exp_dir>/<name>/<ts>/run.json`, `:experiments`
merge + `compare(a, b)` with per-metric deltas);
byproduct UTF-8 string-literal lexer fix; string-valued
variable bindings; three new tutorial lessons. Deferred:
`experiment` source-text capture (needs source threading
through eval), `load` sandbox disallowing absolute-path
canonicalized symlinks (currently path-component-based).

### Saga 13 -- Tiny LM end-to-end (NEXT)
Train a character-level or tiny-BPE language model (~1-5M params)
on a small corpus entirely in MLPL on CPU. Visualize loss,
sample generations, and attention maps. This is the first saga
that proves the platform thesis end-to-end.

### Saga 14 -- MLX backend (COMPLETE, v0.11.0)
Ten steps across five phases. Shipped: `crates/mlpl-mlx` runtime
target with matmul/elementwise/activations/reshape/transpose/
reductions/softmax/log_softmax/cross_entropy each parity-tested
vs CPU within fp32 tolerance; `device("mlx") { body }` scoped
form + `to_device(x, target)` movement helper + typed
`EvalError::DeviceMismatch`; Model DSL dispatch so `apply(model,
X)` inside an MLX block routes every matmul/softmax/add through
`mlpl-mlx`; autograd via tape re-materialization (grad matches
CPU within 1e-4 across every tape primitive); Adam/momentum_sgd/
`train N { }`/`experiment` all inherit; `demos/tiny_lm_mlx.mlpl`
training end-to-end with loss curve matching CPU; Criterion
bench harness (`mlpl-bench --features mlx`). Performance: 0.84x
reshape+reduce, 0.26x tiny_lm_train_step -- below the 5x gate;
bottlenecks diagnosed in `docs/benchmarks.md` (f32/f64
round-trip per op, no graph fusion, tape re-materialization,
small inner dims). Correctness intact. CUDA still Saga 17. See
`docs/milestone-mlx-backend.md`, `docs/using-mlx.md`
retrospective, and `docs/saga.md` Saga 14 entry.

### Saga 15 -- LoRA / QLoRA and quantization
`lora[rank=8] model`, layer-scoped adapters, shared subspaces,
4-bit quantization for the frozen base. Fine-tune the Saga 13
model on a small instruction set.

### Saga 16 -- Embedding visualization and RAG
`embed`, `plot pca/tsne/umap`, 3D scatter via SVG projection,
nearest-neighbor links. RAG pipeline demo over a small corpus.
Builds directly on the existing viz stack.

### Saga 17 -- CUDA backend and distributed execution
CUDA kernels for the fused ops from Saga 14; `run model on
nodes[...]` primitives; device placement syntax. Homelab LAN
training demo.

### Saga 18 -- Distillation, ICL/ICRL, engram memory
Distillation pipelines, trajectory-based updates, engram-style
memory attachments, and multi-model orchestration
(`orchestrate[planner, executor, verifier]`). The research
frontier items from section 6 of `research2.txt`.

### Saga 19 -- LLM-as-tool integration
REST client built-ins, teacher-model distillation workflows,
codegen helpers. Intentionally last in the "core platform"
sequence: secondary to the "build your own model" story.

### Saga 20 -- Perturbation research demos
Express the [Neural Thickets / nt-rs](https://github.com/swcraig/nt-rs)
algorithm (RandOpt-style weight perturbation specialists +
top-K ensembling) as a single MLPL program with a
specialization heatmap output. Adds four small builtins to the
language surface: `clone_model(m)` (deep copy with fresh param
names), `perturb_params(m, family, sigma, seed)` (family-targeted
randn add, where `family` matches the layer-name patterns the
Saga 11/13 model constructors already use internally),
`argtop_k(values, k)` (index-returning companion to the existing
`top_k`), and `scatter(buffer, index, value)` (scalar write into
a rank-1 array, sized to drive `repeat N` accumulation loops).
Headline demo `demos/neural_thicket.mlpl` trains a Saga 13 Tiny
LM as the base, sweeps four perturbation families on MLX, scores
on held-out tokens, runs a top-K ensemble, and renders a
`[family x seed]` heatmap. Stays single-process (no distributed
coordination -- that is Saga 17) and stays on the Tiny LM (no
real pretrained model -- that needs Saga 15+'s checkpoint format
or Saga 19's REST sidecar). Depends on Saga 14 for the variant-
loop speed budget. See `docs/mlpl-for-neural-thickets.md` for
the design sketch and full strawman source.

### Saga 21 -- CLI server + multi-client UI
New `crates/mlpl-serve` binary that exposes a REST + WebSocket
surface over a long-running MLPL interpreter. One server, many
clients: today's web UI wired to call origin (unblocks `:ask`
via a server-side Ollama reverse proxy -- no CORS gymnastics
on the client), `mlpl-repl --connect <host>`, an eventual
ratatui TUI, and an Emacs client that renders SVG in-buffer
without a browser. Session isolation per bearer token;
`--bind 0.0.0.0` requires `--auth required`; proxy allow-list
gates which LLM providers are reachable and server-side env
vars hold their keys. Ships with the CLI visualization
strategy (auto-write SVG to cache dir + print path) so the
terminal REPL stops dumping raw `<svg>` XML. MVP scope: new
crate skeleton, `POST /v1/sessions` + `POST /v1/.../eval`,
`mlpl-repl --connect`. After MVP: visualization storage +
URLs, proxy endpoints, streaming eval via SSE, cancellation.
Desktop GUI wrapper (tauri/wry) and Emacs client land in
follow-up sagas once the server surface is stable. See
`docs/configurations.md` for the configuration matrix, the
CLI-server architecture diagram, the REST API sketch, and the
security posture.

### Saga 22 -- Feasibility checking + resource estimation
Users on limited hardware (laptops, older GPUs) need a way
to sanity-check a planned operation BEFORE running it so
they don't hit OOM, fill disk, or discover a run will take
days. Saga 22 ships four builtins on top of the Model DSL
that answer "will this fit / how long will it take?"
without the user needing to commit to the run.
`estimate_train(model, steps, batch_size, seq_len [,
dtype_bytes]) -> [params, vram_bytes, disk_bytes, flops,
wall_seconds]` is pure-math over the ModelSpec tree:
walks `ModelSpec::params()`, sums parameter counts,
computes VRAM (fwd + grad + Adam moments + activation
heuristic), derives FLOPS per step by node type (Linear,
LinearLora, Embedding, Attention), and divides by a
device-throughput number to get wall-clock.
`calibrate_device() -> gflops` runs a canned 1024x1024
matmul benchmark on the active device, caches the observed
GFLOPS in `env` (CPU vs MLX cached separately), and makes
subsequent `estimate_train` calls honest instead of the
default conservative 50-GFLOPS lower bound.
`hypothetical_model(name) -> ModelSpec` returns a
structural ModelSpec for SmolLM-135M / 360M / 1.7B,
Llama-3.2-1B, Qwen2.5-0.5B so users can ask "how big would
a SmolLM-135M + LoRA(rank=8) fine-tune be on my laptop?"
WITHOUT needing to download the weights first. `apply` on
a hypothetical errors helpfully -- it is an estimation-only
spec. `feasible(estimate_result, [vram_budget, disk_budget,
wall_budget]) -> 0/1` is the guard pattern:
`if feasible(est, [4e9, 10e9, 600]) { train ... }`.
Zeros in the budget mean skip that dimension. Non-goals
(deferred): actual HuggingFace download + safetensors
parsing (separate future saga); f16/bf16 tensor
dtype machinery (what-if only today via `dtype_bytes`
argument); dynamic profiling (the estimator is honest-
approximate at ~2x accuracy); distributed / multi-GPU
estimation (Saga 17); automatic shrinking / recovery
(estimator reports, does not rewrite). CLI-first;
estimator itself works in web but `calibrate_device`
needs CLI. Depends on Saga 11 (Model DSL) + Saga 15
(LoRA freeze set).

## Dependency graph (abbreviated)

```
9 autograd
  |-- 10 optimizers -- 11 model DSL -- 11.5 named axes -- 12 data+tracking -- 13 tiny LM
  |                                                                                |
  |                                                                                +-- 14 MLX
  |                                                                                |    \-- 20 perturbation demos
  |                                                                                +-- 15 LoRA
  |                                                                                +-- 16 viz/RAG
  |                                                                                +-- 17 CUDA/dist
  |                                                                                +-- 18 distill/ICRL
  |                                                                                +-- 21 CLI server
  |                                                                                +-- 19 LLM REST
  |                                                                                +-- 22 feasibility/estimation (also uses 15)
```

Sagas 14-19 + 22 can reorder based on hardware access and
interest; 9-13 (plus the inserted 11.5) must run in order.
Saga 20 needs 14 finished (variant-loop throughput) but is
otherwise free to slot wherever a research-demo cycle makes
sense. Saga 22 was queued ahead of Saga 19 because the
feasibility checker landed value for every existing user
immediately; it shipped as v0.15.0 on 2026-04-24. Saga 19
followed and shipped as v0.16.0 on 2026-04-24. Saga 21
(CLI server) is next, prioritized ahead of the Linux
host move so the live browser demo keeps working without
local MLX once dev moves off Apple Silicon.

## Start next: Saga 21 -- CLI server + multi-client UI (v0.17.0)

Saga 19 (v0.16.0) just shipped `llm_call(url, prompt,
model) -> string` as a CLI-only language-level builtin;
the browser path was deferred because a same-origin
server-side proxy is needed. Saga 21 builds the missing
piece: `crates/mlpl-serve`, a long-running MLPL
interpreter exposed as a REST + WebSocket server, with
multiple thin clients connecting to it. MVP ships the
server skeleton (`POST /v1/sessions`, `POST /v1/sessions/
{id}/eval`, `GET /v1/sessions/{id}/inspect`, `GET /v1/
health`; constant-time bearer-token compare; `--bind
0.0.0.0` requires `--auth required`), the
`mlpl-repl --connect <url>` CLI client, and the CLI
visualization cache strategy (write returned SVGs to
`$MLPL_CACHE_DIR`, print the path instead of raw XML).
Four steps total: skeleton + sessions + eval -> connect
client + inspect endpoint -> viz cache + docs ->
release v0.17.0.

Explicitly deferred to follow-up sagas after the MVP
proves stable: the server-side LLM proxy with allow-list
(unblocks browser `llm_call` -- a careful security
review is wanted before that ships), visualization
storage URLs, Server-Sent-Events streaming eval,
cancellation, persistence across restarts, web UI
re-routing to call origin, ratatui TUI client, Emacs
client, desktop GUI wrapper (tauri / wry).
