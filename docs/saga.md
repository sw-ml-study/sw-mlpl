# Saga

## Saga -1: Repo scaffolding (COMPLETE)
Repo compartmentalization scaffolding.

## Saga 0: Foundation (COMPLETE)
Foundation and contracts.

## Saga 1: Dense tensor substrate v1 (COMPLETE)
Shape, DenseArray, reshape, transpose, indexing.

## Saga 2: Parser and evaluator foundation (COMPLETE)
Lexer, AST, parser with precedence, AST-walking evaluator.

## Saga 3: CLI and REPL v1 (COMPLETE)
Working REPL with all v1 syntax, built-in functions, tracing.

## Saga 4: Structured trace v1 (COMPLETE)
TraceEvent/Trace types, evaluator instrumentation, JSON export.

## Saga 5: Visual web viewer v1 (DEFERRED)
Deferred to post-MVP. MVP uses CLI + JSON trace export.

## MVP
Sagas 0 through 4 complete. MVP ships with REPL + trace export.

## Saga 6: ML foundations (COMPLETE)
Sigmoid, tanh_fn, pow, comparison operators, axis reductions, mean,
array constructors (zeros, ones, fill), and the first end-to-end
logistic regression demo. Delivered v0.2.

## Saga 7: SVG visualization v1 (COMPLETE)
`mlpl-viz` crate, `svg(data, type[, aux])` built-in, diagram types
(scatter, line, bar, heatmap, decision_boundary), `grid()` helper,
high-level analysis helpers (hist, scatter_labeled, loss_curve,
confusion_matrix, boundary_2d), browser REPL inline SVG rendering,
and download button. Delivered v0.3.

## Saga 11: Model DSL (COMPLETE)
Composition primitives for neural-net models: a new `Value::Model`
runtime value, atomic `linear(in, out, seed)` layers,
parameter-free activation layers (`tanh_layer`, `relu_layer`,
`softmax_layer`), sequential `chain(...)` composition,
`residual(block)` skip connections, `rms_norm(dim)` normalization,
and `attention(d_model, heads, seed)` multi-head self-attention
(tape-lowered for `heads=1`, forward-only for `heads>1`). A
`params(model)` walker returns the flat parameter list, and
optimizers now accept a model identifier directly so
`adam(loss, mdl, lr, b1, b2, eps)` trains every weight the model
owns through differentiable `apply(mdl, X)`. The v0.6 `moons_mlp`
and `tiny_mlp` demos were rewritten as one-line `chain(...)`
expressions, and a new `transformer_block.mlpl` demo stacks
`residual(chain(rms_norm, attention))` and `residual(chain(
rms_norm, linear, relu_layer, linear))` twice to train a tiny
2-layer transformer block end-to-end (loss 143.87 -> 1.02 over
100 Adam steps, strictly monotonic). New REPL introspection
commands -- `:vars`, `:models`, `:fns`, `:wsid`, `:describe` --
ship in both `mlpl-repl` and `mlpl-web`, and a new "Model
Composition" tutorial lesson walks from a single `linear` through
a chain MLP to Adam inside `train { }` on the moons dataset.
Delivered v0.7. See `docs/milestone-modeldsl.md`.

## Saga 10: Optimizers + training loop (COMPLETE)
Built proper training infrastructure on top of Saga 9 autograd.
New built-ins: `momentum_sgd(loss, params, lr, beta)` and
`adam(loss, params, lr, b1, b2, eps)` with per-parameter state
held in an `OptimizerState` map on the evaluation environment;
`cosine_schedule(step, total, lr_min, lr_max)` and
`linear_warmup(step, warmup, lr)` pure scalar schedules; and
two non-linear synthetic datasets `moons(seed, n, noise)` and
`circles(seed, n, noise)` returning `[N, 3]` `[x, y, label]`
matrices in the same layout as `blobs`. New `train N { body }`
language construct binds the iteration index to `step` inside
the body and captures each iteration's final value into a
`last_losses` 1-D array, replacing the manual
`repeat { grad; manual update; record loss }` recipe. Two new
demos -- `demos/moons_mlp.mlpl` and `demos/circles_mlp.mlpl` --
train a tanh MLP with `adam` inside `train { }` and render the
decision boundary with `boundary_2d`. New "Optimizers and
Schedules" tutorial lesson added to the web REPL. Delivered v0.6.

## Saga 9: Autograd v1 (COMPLETE)
Reverse-mode autograd as a language primitive. New `mlpl-autograd`
crate provides a tape-based `Tensor` with backward over add, sub,
mul, div, neg, exp, log, relu, tanh, sigmoid, softmax, sum, mean,
transpose, reshape, and matmul, all gradcheck-verified against
finite differences. Parser surface adds `param[shape]` and
`tensor[shape]` constructors, and a new `grad(expr, wrt)` built-in
lifts array expressions onto the tape and returns the gradient
with respect to a tracked parameter. The v0.4 `tiny_mlp` and
`softmax_classifier` demos were ported to use `param + grad`
instead of hand-written backprop, and a new "Automatic
Differentiation" tutorial lesson walks from a scalar minimization
to a one-layer linear regression. Delivered v0.5.

## Saga 8: ML demos (COMPLETE)
Synthetic data primitives (random, randn, argmax, blobs) and
higher-level ML built-ins (softmax, one_hot), plus six demos wired
into the browser REPL: k-means clustering, PCA via power iteration,
a linear softmax classifier, a tiny MLP on XOR-style data, and a
scaled dot-product attention pattern. Tutorial lessons added for
each. Delivered v0.4.

## Saga 11.5: Named Axes and shape introspection (COMPLETE)
Axis-labeled shapes threaded through `Value::Array` via a new
`LabeledShape` type on `mlpl-core`. `label(x, [...])` and `relabel(x,
[...])` primitives, annotation syntax on assignment (`x : [batch,
time, dim] = ...`), and `reshape_labeled(x, dims, labels)` as the
opt-in re-labeling path. Label propagation through elementwise
ops (with one-None-one-Some accepted and mismatches rejected),
matmul (contraction axis validated, outer dims passed through),
reduce/argmax (reduced axis's label dropped), and `map()`
(preserves labels through every math builtin). `reduce_add`,
`reduce_mul`, `argmax`, and `softmax` accept an axis name string
in place of an integer. Structured `EvalError::ShapeMismatch
{ op, expected: LabeledShape, actual: LabeledShape }` at the
evaluator boundary, with Display rendering as `op: expected
[seq=N, d=M], got [time=N, d=M]`. Label-aware `:vars` and
`:describe` (using `LabeledShape` Display), and trace JSON that
round-trips axis labels for labeled arrays and omits the key
entirely for unlabeled ones. New "Named Axes" tutorial lesson in
the web REPL; "Model Composition" lesson now annotates X as
`[batch, feat]` so labels flow through `apply(mdl, X)`. Delivered
v0.7.5. See `docs/milestone-named-axes.md`.

## Saga: Compile-to-Rust (COMPLETE)
Lowers MLPL source to Rust `TokenStream` shared by three targets:
the `mlpl!` proc macro (compile-time embed inside Rust apps), the
`mlpl build foo.mlpl -o bin` subcommand (native-binary and
cross-compile via cargo+rustc, verified for native and
wasm32-unknown-unknown), and a leaner future WASM path that
reuses the same pipeline. New crates: `mlpl-rt` (runtime target;
typed primitive wrappers around `DenseArray`/`LabeledShape`),
`mlpl-lower-rs` (AST -> `TokenStream` with path-configurable
runtime prefix and static matmul contraction checks when both
operands' labels are known at lower time), `mlpl-macro`
(proc-macro wrapper), `mlpl` (user-facing facade with hidden
`__rt` re-export), `mlpl-parity-tests` (parity harness -- nine
curated programs run through both paths and agreement is
asserted bit-for-bit). Static label mismatches surface as
`LowerError::StaticShapeMismatch` which the proc macro converts
to `compile_error!`; the `mlpl-build` CLI surfaces them as
`mlpl-build: ...` prefixed stderr before cargo is even invoked.
Measured 9.05x speedup on a 100x100 reshape+reduce workload
(interpreter 479us -> compiled 53us, median of 5). Deferred:
`TensorCtor` (`param`/`tensor`), `Repeat`, `Train`, autograd
(`grad`), optimizers (`adam`/`momentum_sgd`), and Model DSL
(`chain`/`linear`/activations) are all out of compile scope for
this saga; they require tape-state or loop lowering. Delivered
v0.8. See `docs/milestone-compile-to-rust.md`.

## Saga 21: CLI Server + Multi-Client UI (MVP) (COMPLETE)
First piece of the multi-client story: a long-
running MLPL interpreter exposed as a REST server
(`crates/mlpl-serve`), a thin CLI client
(`mlpl-repl --connect <url>`), and a content-
addressed CLI visualization cache
(`MLPL_CACHE_DIR`) that finally stops the terminal
from dumping raw `<svg>` XML. Three new packages,
four new contracts, one new retrospective doc. The
server skeleton (auth.rs / sessions.rs /
handlers.rs / server.rs / main.rs, each under the
7-fn cap) ships `POST /v1/sessions` (no auth, hands
out a 32-char alphanumeric bearer token),
`POST /v1/sessions/{id}/eval` (auth required;
lex+parse+`eval_program_value` against the
session's `Environment`; returns
`{value, kind}`), `GET /v1/sessions/{id}/inspect`
(auth required; structured workspace snapshot --
vars sorted+capped at 200 with `more` count, models
/ tokenizers / experiments sorted+deduped), and
`GET /v1/health`. Constant-time token compare via
`subtle::ConstantTimeEq`; `--bind 0.0.0.0` (or any
non-loopback) refuses to start under
`--auth disabled`. axum 0.7 + tokio + tower stack.
The `mlpl-repl --connect <url>` client splits
between `connect.rs` (HTTP transport: 4 fns) and
`connect_repl.rs` (REPL loop + slash dispatch +
`--connect` argv parser: 4 fns) -- two modules
because connect.rs would have hit the 7-fn cap
otherwise. Local `Environment` is unused in connect
mode; slash commands `:vars` / `:models` /
`:tokenizers` / `:experiments` / `:wsid` go through
`/inspect`, while `:ask` keeps using
`mlpl_runtime::call_ollama` directly (server-side
framing follows in a future saga). `--connect` is
incompatible with `-f` / `--data-dir` / `--exp-dir`
(parser errors with exit code 2). The CLI viz
cache lives at `crates/mlpl-cli/src/viz_cache.rs`:
SHA-256-prefix-12 chars under
`$MLPL_CACHE_DIR` / `dirs::cache_dir().join("mlpl")`,
with content-addressed naming so identical viz
outputs don't duplicate; non-SVG values pass
through. Wired into both local `eval_line` and
connect-mode `read_loop`. `svg_out.rs` simplified
from the previous SVG handler to a thin holder for
the `--svg-out <dir>` back-compat override flag.
35 new tests across the saga (13 server +
4 connect + 18 viz_cache); sw-checklist baseline
139 fails held throughout. Hit two budget walls
during impl -- connect.rs at 8 fns split into
connect.rs + connect_repl.rs; main.rs's `main` fn
at 58 LOC compressed via a local closure for the
flag-lookup pattern (couldn't extract a top-level
helper because main.rs was already at the 7-fn
cap). `docs/using-cli-server.md` is the user-
facing retrospective; `docs/configurations.md`
refreshed with new rows + footnote [8] explicitly
distinguishing "MVP shipped" from "LLM-proxy
follow-up still deferred." Carved out as
post-MVP and listed in the deferred non-goals:
the server-side LLM proxy with allow-list
(unblocks browser `llm_call`; needs a security
review for allow-list + secret handling first),
visualization storage URLs, Server-Sent-Events
streaming, cancellation, persistence across
restarts, web UI re-routing to call origin,
reattach across REPL restarts, the WebSocket
surface, the desktop GUI wrapper (tauri / wry),
the Emacs client, the ratatui TUI client, and
other viz formats (PNG / HTML / JSON). Each lands
in a follow-up saga after the MVP server contract
proves stable. Delivered v0.17.0. Next: the dev
host move to Linux + NVIDIA, then Saga 17 (CUDA +
distributed execution).

## Saga 19: LLM-as-Tool REST Integration (COMPLETE)
A language-level builtin that POSTs to an Ollama-
compatible `/api/generate` endpoint and returns the
model's completion text as a `Value::Str` scalar.
The REPL's preview `:ask` command (shipped in
v0.11.x as the Saga 19 preview) predated this; it
now delegates to the same `mlpl_runtime::call_ollama`
path so the HTTP machinery lives in one place. The
headline use case is "LLM as a tool" -- pipe a value
through a hosted LLM, store the reply as an MLPL
string, feed it to `tokenize_bytes` /
`experiment {}` / the rest of the pipeline. The
language surface is exactly `llm_call(url, prompt,
model) -> string`: URL is normalized (trailing
slashes stripped, `/api/generate` appended unless
already present, no double-append), 120-second
timeout matches the `:ask` preview, returns the
top-level `response` field as a string. Non-2xx
status, invalid JSON, or missing `response` field
all surface as actionable errors carrying
`"llm_call"`, the URL, and a 200-character body
preview. CLI-only -- the browser cannot reach
localhost Ollama without CORS allow-listing + a
server-side proxy; that path lands in Saga 21
(`mlpl-serve`). Module layout splits the pure HTTP
path (`crates/mlpl-runtime/src/llm_builtins.rs`, 3
fns) from the `Expr`-aware dispatch shim
(`crates/mlpl-eval/src/llm_dispatch.rs`, 1 fn) --
mirrors Saga 22's revised layout. **Design
deviation** from the original step 001 plan, which
proposed 5 functions in the runtime crate
(try_call / builtin_llm_call / validate_args /
resolve_url / parse_response): try_call and
validate_args need `Expr` access that the runtime
crate intentionally avoids, so they moved to the
eval-side dispatcher. The :ask migration is a
**behavior change**: the pre-v0.16 :ask used
Ollama's `/api/chat` with explicit
`{role: system}` + `{role: user}` messages; the
new :ask concatenates the same content into a
single prompt and POSTs to `/api/generate`. Model
loses the role distinction but keeps the context;
TODO note in `apps/mlpl-repl/src/ask.rs` flags a
future `llm_chat(history, prompt)` variant (listed
in step 001's deferred non-goals) as the path to
restore `/api/chat` semantics. 8 mockito-based
integration tests pin the URL-resolution +
arity + error catalog. New `demos/llm_tool.mlpl`
exercises `llm_call` -> `tokenize_bytes`
end-to-end (verified live during step 002 against
a running local Ollama). `docs/using-llm-tool.md`
is the user guide; `docs/using-ollama.md` was
updated to redirect to it; `docs/configurations.md`
gained an `llm_call` row + footnote [7]. Streaming
SSE, OpenAI-style tool calling, multi-turn chat,
request batching, in-source auth secrets,
teacher-model distillation pipelines, and
web / WASM support all stay deferred -- see the
contract for the full list. sw-checklist baseline
102/139 held across both feature steps. Delivered
v0.16.0. Saga 21 (CLI server) is next.

## Saga 22: Feasibility Checking + Resource Estimation (COMPLETE)
Four new language-level builtins that let the user
sanity-check a planned training run BEFORE committing
disk, RAM, and hours of wall-clock to it. Pure math
over a `ModelSpec` (or a hardcoded HF-scale dimension
table) plus train-loop parameters; no weights required
for what-if queries. `estimate_train(model, steps,
batch_size, seq_len [, dtype_bytes])` returns a
rank-1 `[5]` f64 array `[params, vram_bytes,
disk_bytes, flops, wall_seconds]` with labels pinned
in the contract -- VRAM sums forward weights +
gradient + Adam m/v moments (trainable only; LoRA
frozen base is zero on the grad/adam legs) + a `4x`
activation safety factor; FLOPS covers Linear /
LinearLora / Attention / Embedding; wall-clock reads
throughput from a cached env key defaulting to 50
GFLOPS (CPU laptop lower bound). `calibrate_device()`
runs a 10-iteration 1024x1024 matmul benchmark
through `device::dispatched_call` (first iter is
warmup), measures wall-clock, writes observed GFLOPS
into `mlpl_device_throughput_gflops`; device-aware
under `device("mlx") { }` so subsequent estimates on
that device read honest numbers.
`estimate_hypothetical(name, ...)` talks about
SmolLM-135M / 360M / 1.7B / Llama-3.2-1B /
Qwen-2.5-0.5B via a hardcoded dimension table --
**design deviation from the original plan**, which
proposed `hypothetical_model(name) -> ModelSpec` that
`estimate_train` would walk. That path required
materializing zero arrays in `env` for every
parameter in the spec, and a SmolLM-1.7B query would
have cost ~14 GB of RAM just to ask the question.
Shipped as a direct `[5]`-return builtin instead; the
output shape is identical to `estimate_train` so
`feasible(...)` composes unchanged.
`feasible(estimate_result, budget)` takes the `[5]`
estimate plus a `[3]` `[vram, disk, wall]` budget
(zeros skip that dimension) and returns scalar 1/0
for `if feasible(est, [4e9, 1e10, 600]) { train ... }`
guards. 23 new tests (11 estimate + 12 feasibility;
one ignored for the slow default-size calibrate
bench). Two contracts for step 001
(`contracts/eval-contract/estimate.md`) plus three
for step 002 (`calibrate-device.md`,
`estimate-hypothetical.md`, `feasible.md`). Module
layout: `crates/mlpl-runtime/src/estimate_builtins.rs`
(6 fns) + `crates/mlpl-eval/src/model_feasibility.rs`
(7 fns at limit); sw-checklist baseline 102/139 held
across both steps. `demos/feasibility.mlpl` tours
all four builtins end-to-end (CLI-only because
browser timers are too noisy for calibration).
`docs/using-feasibility.md` retrospective + user
guide documents the estimator math, the ~2x
accuracy target, the hardcoded HF-scale table, the
LoRA-on-SmolLM worked example, the guard pattern,
and the deferred non-goals (HF download +
safetensors loading, f16/bf16 tensor support,
dynamic profiling, distributed / multi-GPU, auto-
recovery). Delivered v0.15.0. Saga 19 (LLM-as-tool
REST `llm_call`) is next.

## Saga 16: Embedding Visualization (COMPLETE)
Embedding-inspection surface for any model with an
`embed` layer or any rank-2 `[N, D]` intermediate
representation. Three new builtins plus one new viz
type close the "train a model, inspect what it
learned" loop: `pairwise_sqdist(X)` returns the
`[N, N]` squared-Euclidean distance matrix with
symmetry + zero-diagonal + empty-input safety;
`knn(X, k)` returns each row's k nearest non-self
neighbors sorted by ascending distance with lower-
index tie-break (self never appears in its own row);
`tsne(X, perplexity, iters, seed)` runs classic van
der Maaten t-SNE to produce `[N, 2]` with per-row
perplexity-calibrated binary search, symmetrized
joint probabilities, Student-t low-dim affinities,
momentum-based KL-gradient descent with early
exaggeration, deterministic under identical seeds.
`svg(pts, "scatter3d")` renders `[N, 3]` (point
cloud) or `[N, 4]` (points + integer cluster id) as
a static orthographic-projection SVG with labeled
axis gizmos + one `<circle>` per point + optional
sorted legend. `demos/embedding_viz.mlpl` composes
the pipeline end-to-end: train a standalone
`embed(12, 8, 0)` toward a structured 3-cluster
target via MSE for 50 adam steps; extract the
learned table via `apply(emb, iota(V))`; render
through t-SNE + column-selector + `knn` to show that
learned embeddings recover the target structure
(every token's nearest neighbors stay in its own
cluster). 35 tests across five files pin every
invariant. The t-SNE builtin was first written as a
300-line monolith (three new sw-checklist FAILs);
user feedback paused the work to document the
decomposition rules, producing
`docs/sw-checklist-patterns.md` (catalog of struct-
return, struct-args, validate-then-work, orchestrator
+ helpers, phase helpers, extract-to-module, chain-
of-responsibility, bridge, split-tests-from-work
patterns with worked examples from this codebase).
The monolith was then split into four sibling modules
(`tsne_builtin` / `tsne_validate` / `tsne_affinities`
/ `tsne_gradient`), each a single-responsibility
facade under every budget; baseline 139/102 sw-
checklist held. PCA stays a composition pattern (the
Saga 8 power-iteration + deflation lesson) rather
than shipping a new builtin. t-SNE itself is CPU-only
(inner loop does not vectorize cleanly through MLX).
UMAP, RAG pipeline (pending Saga 19), interactive
3-D, MLX for tsne, Barnes-Hut approximate t-SNE all
deferred; see `docs/using-embeddings.md` "Not
shipped". Delivered v0.14.0.

### Addendum: Saga 16.5 -- Embedding-viz Polish (COMPLETE)

Two convenience builtins that closed the Saga 16 loose
ends flagged in the shipped `docs/using-embeddings.md`
"Not shipped" list: `pca(X, k) -> Y [N, k]` wraps
power iteration + Gram-Schmidt + deflation into a
one-liner (the Gram-Schmidt-inside-power-iteration
step is load-bearing for `k = D` on rank-deficient
inputs where plain deflation lets `v` drift back
toward earlier components), and `embed_table(model) ->
[vocab, d_model]` walks a `ModelSpec` tree depth-
first left-to-right and returns the first Embedding
layer's lookup table cloned from `env` (first-match
semantics; Chain / Residual / nested chain all
traverse; no-Embedding errors cleanly). 17 new tests
(9 pca + 8 embed_table), both modules under the 7-fn
budget by design (6 and 2 respectively), sw-checklist
baseline 139/102 held. `demos/embedding_viz.mlpl`
now uses `pca(table, 3)` for the 3-D projection
(principled instead of the first-three-dims selector
shortcut). `docs/using-embeddings.md` rewrites the
PCA section from "composition-only" to "shipped as a
builtin in v0.14.1" (keeps the power-iteration recipe
as pedagogy), adds an "Extracting embed-layer
weights" section, and updates the "Training inside a
chain" flow to use `embed_table(m)` instead of the
`apply(standalone_embed, iota(V))` workaround. Web
REPL "Embedding exploration" lesson adds `pca(X, 2)`
and `embed_table(emb)` examples. Path-selector
variants of `embed_table` (e.g., `embed_table(m,
"encoder.embed")`), multi-embedding stacks, and
streaming / randomized / kernel PCA all deferred.
Delivered v0.14.1. In-family polish, not a full saga.

## Saga 15: LoRA Fine-Tuning (COMPLETE)
Parameter-efficient fine-tuning for the Model DSL. Three new
builtins compose into a PyTorch-`peft`-style "frozen base,
trainable low-rank adapters" workflow: `freeze(m)` /
`unfreeze(m)` mark every `m.params()` name in a new
`env.frozen_params` set that `adam` and `momentum_sgd` skip
at the optimizer-update stage (gradient computation is
unchanged, so the chain rule still flows through frozen
params); `lora(m, rank, alpha, seed)` clones `m`'s spec tree,
replaces every `Linear` node with a new `ModelSpec::LinearLora`
that owns the cloned base `W`, `b` plus two fresh adapter
matrices `A [in, rank]` (init `randn * 1/sqrt(in)`) and
`B [rank, out]` (init zeros), and -- after step 004's scope
amendment -- auto-freezes EVERY non-adapter param in the
student (embed tables, attention projections, and MLP base
linears all freeze consistently). The zero-init on B is the
standard LoRA "forward identity before training" property:
`apply(lora_m, X) == apply(m, X)` elementwise before any
gradient step. Forward formula `y = X @ W + (alpha / rank) *
X @ A @ B + b` composes through existing matmul / scalar-mul
/ add dispatch, so MLX (Saga 14) picks it up for free -- no
new tape ops were needed. `demos/lora_finetune.mlpl` trains a
Saga 13 Tiny LM on Shakespeare for 100 steps, wraps with
rank-8 LoRA, fine-tunes adapters on a synthetic Q/A
instruction corpus for 50 steps; base bit-identical
throughout, final fine-tune cross-entropy ~2.18.
`demos/lora_finetune_mlx.mlpl` is the Apple-Silicon variant
with the fine-tune loop under `device("mlx") { }`. A new
Criterion bench group `lora_finetune_step` measured warm
CPU 164us vs warm MLX 1.11ms = 0.15x (MLX slower at this
scale -- LoRA doubles the matmul count per linear and the
rank-2 adapter matmuls are too small to amortize MLX's
kernel-launch overhead; at d=512 the ratio would flip).
CPU-vs-MLX parity across fine-tune losses + every student
param (both adapters and frozen base) within 1e-3, pinned
by `crates/mlpl-eval/tests/lora_mlx_demo_tests.rs`. A
"LoRA Fine-Tuning" web REPL tutorial lesson (tiny V=8, d=4,
rank=2 variant) runs interactively in WASM. Delivered
v0.13.0. See `docs/using-lora.md` and `docs/benchmarks.md`.

## Saga 20: Perturbation research demos / Neural Thickets (COMPLETE)
Four new builtins plus a headline demo that compose MLPL into
a RandOpt-style weight-perturbation workflow. Seven steps in
three phases. Phase 1 shipped the language surface: `clone_model(m)`
deep-copies a `ModelSpec` tree and allocates a disjoint set of
fresh param names while preserving device tags (step 001);
`perturb_params(m, family, sigma, seed)` walks a model's
params, filters by one of four families (`all_layers`,
`attention_only`, `mlp_only`, `embed_and_head`), and adds
`sigma * randn(seed + i, shape)` to each matching param in
place, with the "final projection head" detected structurally
(the last top-level `Linear` child of the outermost `Chain`,
not a name-pattern heuristic) so `mlp_only` / `embed_and_head`
can tell MLP linears apart from the vocab projection (step
002); `argtop_k(values, k)` returns the k indices of the
largest entries in a rank-1 vector (descending by value, ties
to lower index; stable sort) and `scatter(buffer, index,
value)` returns a copy of a rank-1 buffer with the single
entry at `index` replaced by `value` (step 003, both in a
new `crates/mlpl-runtime/src/ensemble_builtins.rs`). Phase 2
shipped `demos/neural_thicket.mlpl` -- train a Saga 13 Tiny
LM base, sweep 4 families x 4 seeds = 16 variants, score
each on a held-out string, render a `[family x seed]`
specialization heatmap, pick top-K via `argtop_k`, and
average logits across all 16 variants for an ensemble
cross-entropy (step 004, CPU-only; uses `for i in [0,1,2,3]`
loops per family because `repeat` does not expose a counter
and MLPL lacks string-array indexing for `families[i/4]`).
`demos/neural_thicket_mlx.mlpl` wraps the variant loop +
ensemble in a single `device("mlx") { ... }` block (base
training stays on CPU); losses and ensemble logits agree
with the CPU path elementwise within fp32 tolerance, pinned
by a triple-gated integration test (step 005). A new
`mlpl-bench` Criterion group `neural_thicket_variant_loop`
measures warm CPU 767us vs warm MLX 3.01ms (MLX 0.25x), a
ratio essentially identical to Saga 14's 0.26x on
`tiny_lm_train_step` -- evidence that the bottleneck at
Tiny LM inner dimensions is per-op kernel launch + f32
round-trip, not autograd-specific cost. Phase 3 shipped a
"Neural Thickets" web REPL tutorial lesson (tiny 4x4 sweep
at V=8/d=4 so the full heatmap renders in the browser),
`docs/using-perturbation.md` (retrospective covering the
four builtins, the four families, the heatmap narrative,
measured MLX numbers, and the deferred follow-up surface:
depth-aware families / low-rank perturbation / real
checkpoints / strict top-K ensembling / `repeat`
step-counter), and rebuilt `pages/` for the live demo
(step 006). Delivered v0.12.0. See
`docs/mlpl-for-neural-thickets.md` for the design sketch and
`docs/using-perturbation.md` for the shipped-surface doc.

## Saga 14: MLX backend (COMPLETE)
First accelerator backend for MLPL: a program that trains on CPU
now runs on Apple Silicon via MLX without source changes. Ten
steps across five phases. Phase 1 built a `crates/mlpl-mlx`
sibling to `mlpl-rt` with MLX-backed equivalents of every
forward primitive the Tiny LM touches (matmul, add/sub/mul/div/
neg, exp/log/tanh/sigmoid/relu, reshape/transpose, reduce_mul/
mean/argmax axis-aware, softmax/log_softmax, fused
cross_entropy), each parity-tested vs CPU within documented
fp32 tolerance. Phase 2 added the `device("mlx") { body }`
scoped form in the parser/AST (mirroring `experiment`'s shape),
the `to_device(x, target)` movement helper (also walks the
param tree for models), cross-device-safety as a typed
`EvalError::DeviceMismatch { op, expected, actual }` variant,
and routed every Saga 11 model's forward (`linear`, `chain`,
`residual`, `rms_norm`, `attention`, `causal_attention`,
`embed`, activations) through a single `dispatched_call`
helper. Phase 3 wired autograd on MLX: `grad(expr, wrt)` inside
a `device("mlx") { }` block re-materializes the CPU-built tape's
forward values through `mlpl-mlx` before CPU backward formulas
compute gradients, producing gradients that match the all-CPU
path within fp32 tolerance across every tape primitive (sum,
mean, exp, log, relu, tanh, sigmoid, neg, add, mul, div,
softmax, transpose, matmul) plus the full `grad(cross_entropy(
apply(model, X), Y), W)` integration path. Adam, momentum_sgd,
`train N { body }`, `last_losses`, `cosine_schedule`,
`linear_warmup`, and `experiment "name" { body }` all inherit
from the grad wiring and produce parameter updates and loss
curves that match CPU on a Tiny LM-shaped slice within 1e-4.
Phase 4 shipped `demos/tiny_lm_mlx.mlpl` (the Saga 13 body
wrapped in `device("mlx") { }`, identical seeds, hyperparams,
dataset, 200 steps, loss curve matches CPU) and extended
`mlpl-bench` with a Criterion harness comparing CPU vs MLX on
`reshape_reduce_100x100` and a `tiny_lm_train_step` workload,
reporting both cold and warm wall-clock timings. Phase 5 added
a "Running on MLX" web REPL tutorial lesson, turned
`docs/using-mlx.md` from design sketch into shipped-API
reference with a retrospective, and rebuilt the live demo page.
Measured performance at Tiny LM scale: MLX is 0.84x on
reshape+reduce and 0.26x on the training step -- below the saga
plan's 5x warm-path gate. Four bottlenecks diagnosed in
`docs/benchmarks.md` (f32/f64 round-trip per op, no graph
fusion, tape re-materialization doubles forward work, d=16/32
inner dims dominated by kernel-launch overhead); a dedicated
"MLX throughput" follow-up step is the natural next work item.
Correctness is intact -- every MLX-gated parity test across
the saga passes. Delivered v0.11.0. See
`docs/milestone-mlx-backend.md` and `docs/using-mlx.md`.

## Saga 13: Tiny language model end-to-end (COMPLETE)
First saga that proves the platform thesis: a few lines of MLPL
train a small transformer LM, generate text from a prompt, and
render the attention pattern -- all on CPU, all reproducible via
experiment tracking. Six new primitives filled the gap between
Saga 12's tokenizer surface and a working LM: `embed(vocab,
d_model, seed)` as a `Value::Model` with a learned `[vocab,
d_model]` table and gradient flow; deterministic
`sinusoidal_encoding(T, d_model)` additive positional tables;
`causal_attention(d_model, heads, seed)` with a lower-triangular
pre-softmax mask so position `t` cannot peek at `t+1`;
numerically-stable `cross_entropy(logits, targets)` over integer
targets with max-subtraction log-softmax; `sample(logits, t,
seed)` multinomial draws plus `top_k(logits, k)` pre-softmax
restriction; and a `last_row` / `concat` / `attention_weights`
triple that makes a generation loop and a `[T, T]` attention
heatmap one-liner. New demos `demos/tiny_lm.mlpl` (end-to-end
training) and `demos/tiny_lm_generate.mlpl` (generation +
heatmap) wire every Saga 13 primitive together with Saga 12's
BPE tokenizer, `shift_pairs_x`/`shift_pairs_y` next-token pair
windows, and `experiment "name" { }`-tracked `train N { adam(... )
}`. Two new web REPL tutorial lessons -- "Language Model Basics"
(forward-only, runs in <2s) and "Training and Generating"
(stripped-down training loop + 20-token generation + attention
heatmap) -- ship alongside. Delivered v0.10.0. See
`docs/milestone-tiny-lm.md`.

## Saga 12: Tokenizers, datasets, and experiment tracking (COMPLETE)
Closes the last surface-only gap before the Tiny LM. File IO:
`load("rel.csv")` / `load("rel.txt")` reads through an
`Environment::data_dir` sandbox (terminal REPL `--data-dir`
flag); `load_preloaded("name")` serves compiled-in corpora for
the web REPL. Dataset prep: `shuffle(x, seed)` Fisher-Yates row
permutation, `batch(x, size)` with zero-padded short batches +
matching `batch_mask`, `split(x, frac, seed)` and
`val_split(x, frac, seed)` returning disjoint chunks; `for row
in ds { body }` streaming iteration with `last_rows` capture
(new `Token::For`/`Token::In` keywords, `Expr::For` AST
variant). Byte-level BPE: `tokenize_bytes` / `decode_bytes`
primitives; `Value::Tokenizer` runtime variant with
`TokenizerSpec::ByteLevel` and `TokenizerSpec::BpeMerges`;
`train_bpe(corpus, vocab_size, seed)` with deterministic
lex-smallest tie-breaking; `apply_tokenizer(tok, text)` +
`decode(tok, tokens)` byte-lossless round-trip for any UTF-8
input. Experiment tracking: `experiment "name" { body }` scoped
form captures `_metric`-suffixed scalars and bound params'
shapes into `ExperimentRecord`s, logged to
`env.experiment_log` and (terminal REPL `--exp-dir`)
`<dir>/<name>/<ts>/run.json` via serde_json; `:experiments`
REPL command merges memory + disk; `compare(a, b)` builtin
returns side-by-side per-metric deltas. Byproduct: lexer UTF-8
fix so string literals now carry arbitrary Unicode code points
(previously was `b as char` Latin-1 mojibake). Three new
tutorial lessons in the web REPL ("Loading Data", "Tokenizing
Text", "Experiments"). String-valued variable bindings
(`corpus = load_preloaded("...")`) via new
`Environment::strings` map. Delivered v0.9.0. See
`docs/milestone-tokenizers.md`.
