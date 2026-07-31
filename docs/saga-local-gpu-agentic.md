# Saga: Local-GPU fine-tuning + agentic, tool-using `:ask`

## Vision

Reach the workflow where a user, inside the sw-MLPL REPL, can:

1. Fine-tune a small but REAL local model on the GPU (MLX on Apple
   Silicon, CUDA on Linux) and SEE measurable learning happen.
2. Ask Ollama for help with that fine-tuning -- an agentic,
   tool-using `:ask` that REQUESTS the training context it needs
   (loss curve, hyperparameters, model shape) and answers from what
   it retrieved (Recursive-Language-Model-style tooling).

This merges the previously-planned agentic-`:ask` work
(`docs/agentic-ask-plan.md`) with a new "Local GPU" demo group.

## Foundation already in place

- **Connect mode (SSE).** `?connect=<url>` routes REPL eval to
  `mlpl-serve`, executed server-side. `connect.rs` +
  `eval_wasm::connect_eval`.
- **Contextual `:ask` + real system role (shipped).** `llm_call`
  takes an optional 4th `system` arg -> Ollama's `system` field;
  web `:ask` sends the question as the user prompt and the sw-MLPL
  grounding + REPL history + selected-sculpture context as the
  system message. Plus `:history`, the in-dialog Ask button,
  `?ollama=` / `?model=`.
- **LoRA fine-tuning of MLPL-native models.** `lora(m, rank,
  alpha, seed)` wraps every `Linear` with adapters; `adam(loss,
  lora_m, ...)` updates ONLY the adapters. MLX device support
  (`device("mlx")`, `mlpl-mlx-rt`, `mlpl-mlx-serve`). Working demos:
  `demos/lora_finetune.mlpl`, `demos/lora_finetune_mlx.mlpl`.

## Policy decisions (load-bearing -- bake into every step)

### Model acquisition: demo-initiated, allow-listed, never silent

The headline demo DOWNLOADS the pretrained model as its first step
(measure -> fine-tune -> re-measure -> show improvement). That is
allowed because the user PERMITS it by choosing to run the demo --
it is not a silent or background fetch. The guardrails that keep it
inside the "no large downloads without permission" rule:

- **Named + allow-listed, never an arbitrary URL.** A narrow builtin
  (e.g. `fetch_pretrained("smollm2-135m")`) maps a SHORT allow-listed
  NAME to a pinned source + cache path. There is NO `download(url)`.
- **Size disclosed up front.** The demo states the download size
  (e.g. "~270MB SmolLM2-135M, one time, cached") before fetching, and
  skips re-download if already cached.
- **Server-side in connect mode.** The browser cannot write the
  user's disk; in connect mode `mlpl-serve` performs the allow-listed
  fetch + cache. The download is therefore connect-only (the public
  GitHub Pages demo marks it visible-but-not-runnable).
- **CLI mirrors it.** The CLI demo runs the same allow-listed fetch
  locally (or reads an already-pulled model by path), then the same
  measure / fine-tune / re-measure arc.

Ollama-served inference still uses `ollama pull smollm2:135m`; the
FINE-TUNING path needs raw weights (safetensors/bf16), which is what
`fetch_pretrained` caches.

### External IO: narrow capabilities, no general shell

No general `sh()` / `run()` / `curl` escape. The web playground
runs UNTRUSTED pasted input and the server runs `--auth disabled`
on loopback -- arbitrary shell there is remote code execution.
Pattern: narrow, purpose-built, allow-listed builtins (like the
existing `llm_call`). Web/WASM builds get NO shell and NO arbitrary
network. Any future shell-like capability is native-CLI-only, off
by default behind an explicit `--allow-shell` flag, never compiled
into WASM.

## Demo capability tiers (device gating)

Demos declare a capability tier; the UI gates them by it. MLX
(Apple GPU) and CUDA (NVIDIA/Linux GPU) are SEPARATE groups; both
are connect-only + device-specific and NOT runnable on the public
GitHub Pages live demo. CPU-based training IS runnable on the live
demo (the in-browser WASM interpreter runs it, slowly).

| Tier | Runs on live demo? | Needs | Examples |
|------|--------------------|-------|----------|
| `cpu` | YES (in-browser WASM) | nothing | tiny LoRA fine-tune (the 6.12 -> 1.34 run), core playground |
| `connect` | NO (needs a server) | `mlpl-serve` | contextual `:ask` (Ollama), `:models ollama` |
| `mlx` | NO | `mlpl-serve` + Mac MLX peer | MLX-accelerated demos (Apple Silicon) |
| `cuda` | NO | `mlpl-serve --features cuda` (Linux + NVIDIA) | CUDA LoRA fine-tune (live; see saga `cuda-foundation`) |

Registry flag shape (Phase 1): each demo carries
`{ requires_connect: bool, device: cpu|mlx|cuda }`. The public
build renders `connect`/`mlx`/`cuda` demos visible-but-not-runnable
with a "needs a connected mlpl-serve (+ <device> peer)" affordance;
`cpu` demos run everywhere.

## RESOLVED (step 010, 2026-05-31): MLX LoRA training is now full-GPU

The limitation described below is FIXED for the LoRA fine-tune path.
Under `device("mlx")`, `eval_adam` now detects a head-only LoRA model
(the demo architecture) and runs the whole fine-tune step on the GPU:
the loss is built as an `mlx_rs`-traceable graph over the adapters
(`mlpl-mlx-model::demo_forward`), differentiated by `value_and_grad`,
and the adapters are updated by an MLX-resident Adam (`adam_update`),
with moments persisted in `env.optim_state` across steps. Any other
model/device falls back to the CPU tape path. Parity-tested vs CPU in
`lora_mlx_demo_tests.rs` (loss curve within fp32 tol; frozen base
bit-identical). The `MLX LoRA fine-tune` demo is relabeled true-GPU.
General (non-LoRA) `device("mlx")` training still uses the CPU tape;
the section below documents that original state.

## CRITICAL reality: MLX training was forward-only on the GPU (pre-step-010)

Server-side MLX does NOT yet run a full training loop on the GPU.
With `device("mlx")`, only the FORWARD pass (matmul, attention,
softmax, cross-entropy) dispatches to MLX via `mlpl-mlx-rt`; the
BACKWARD pass (autograd in `mlpl-autograd`) and the adam optimizer
updates (`mlpl-eval/src/grad_optim.rs`, plain `Vec<f64>` math) run
on CPU. `mlpl-mlx-rt` exports forward ops only -- no MLX
backward/gradient/optimizer. So `device("mlx") { train { adam } }`
is a HYBRID (forward on GPU, backward+update on CPU), confirmed by
`grad_mlx_tests.rs` / `tiny_lm_mlx_demo_tests.rs` (parity, not GPU
execution of the backward/update).

Consequence for the plan:

- A NEW phase is required to make the full fine-tune loop run on the
  GPU: implement MLX backward + an MLX-resident adam (extend
  `mlpl-mlx-rt` + the autograd/optimizer path so gradients and
  moment buffers stay on-device, no per-step CPU round-trip).
- Until that lands, the MLX demo must be HONESTLY labeled as
  "forward pass on GPU; gradient + optimizer on CPU" -- not
  "training runs on the GPU".

### Step 004 scoping (2026-05-31): code map + the architecture fork

A read-through of the relevant crates (no build) pins the work:

- **`mlpl-mlx-rt`** (`components/native-rt/crates/`): 7 modules of
  FORWARD ops only (matmul, add/sub/mul/div/neg, exp/log/relu/sigmoid/
  tanh, softmax/log_softmax/cross_entropy/mean/reduce_mul/argmax,
  transpose/reshape). MLX arrays are `mlx_rs::Array`; ops are free
  functions; everything triple-gated
  `cfg(all(target_os="macos", target_arch="aarch64", feature="mlx"))`.
  No backward/gradient/optimizer op exists.
- **Forward dispatch** lives in `mlpl-eval/src/device.rs`
  (`try_mlx_dispatch`, `dispatched_call`): only fires when
  `env.device()=="mlx"`. Device is a `String` on `Environment`'s
  device stack -- it does NOT travel on the tensor/tape node level.
- **`mlpl-autograd`**: the tape is CPU f64; `backward.rs` has 19
  functions and carries NO device info. The MLX hook
  (`materialize_tape_on_mlx`) only round-trips forward VALUES to fp32
  so CPU backward matches MLX within tolerance -- backward itself is
  always CPU.
- **adam** (`mlpl-eval/src/grad_optim.rs`): moment buffers are
  `Vec<f64>` in `Environment::optim_state`; every step calls
  `eval_grad` (CPU tape build + backward) and does the update in f64.
  A full per-step CPU round-trip.

**Architecture fork (decide in step 005 BEFORE implementing):**

- **A) MLX built-in autodiff (recommended).** Express the MLX
  fine-tune step's forward as an `mlx_rs`-traceable function and use
  `value_and_grad`/`grad` for on-device gradients; keep moment buffers
  as `mlx_rs::Array`. Bypasses the 19 hand-written backward formulas
  for the MLX path -- far less code, no parity drift -- at the cost of
  restructuring the MLX loop to build an MLX graph rather than the CPU
  tape.
- **B) Hand-port backward to MLX.** Add MLX backward ops to
  `mlpl-mlx-rt` and make `mlpl-autograd` dispatch backward to MLX when
  `device=="mlx"`, plus an on-device adam. Keeps the tape architecture
  but is ~19 formulas to port and parity-maintain.

**sw-checklist note:** `backward.rs` (19 fns) and `mlpl-mlx-rt`
(`reductions.rs` 7 fns; 7 modules) are already over budget. Splitting
`backward.rs` into sibling files would push the crate past the
7-module FAIL line -- trading one FAIL for another. The correct fix is
a sibling crate (e.g. `mlpl-autograd-backward`) inside the autograd
component with dependency inversion (`Tensor::backward()` must not
re-enter), which is its own step, not incidental paydown.

**Decomposition:** step 004 = scope + decompose + disk recovery (the
shared `target/` had grown to 40 GB with 24 GiB free; cleared to 63
GiB free). Step **005 (`mlx-fullgpu-architecture-spike`)** resolves
the fork with a small spike build and defines the implementation
follow-ons (approach A: traceable-forward then value_and_grad+adam;
approach B: mlx-rt backward -> autograd dispatch -> on-device adam),
each parity-tested vs CPU within fp32 tolerance, ending with the
demo relabeled hybrid -> true-GPU.

### Step 005 result (2026-05-31): approach A DECIDED + spike proven

**Decision: approach A** (MLX built-in autodiff). The vendored
`mlx-rs` (0.25.3) exposes `transforms::value_and_grad` /
`value_and_grad_with_argnums` (closure `|&[Array]| -> Result<Vec<Array>>`
-> `(values, grads)`) AND an `optimizers` module (Adam/AdamW/Adamax).
So the MLX fine-tune step can be differentiated on-device with zero
hand-written backward formulas -- approach B (porting ~19 formulas) is
abandoned.

**Spike (proven, parity-tested):** new sibling crate
`components/native-rt/crates/mlpl-mlx-train` ("refactor up and out":
training is a distinct concern from the forward-only `mlpl-mlx-rt`,
so its own crate; native-rt now has 3 crates, within the <=4 budget;
sw-checklist 6 passed / 0 failed / 0 warnings). It provides:

- `loss_and_grads(params, loss_fn)` -- wraps `value_and_grad_with_argnums`
  over all params; returns the scalar loss + one gradient `Array` each.
- `MlxAdam` -- Adam whose m/v moment buffers ARE `mlx_rs::Array`; the
  update is pure MLX elementwise ops, so gradients and optimizer state
  never leave the device.

Two gated tests pass on Apple Silicon (`cargo test -p mlpl-mlx-train
--features mlx`): `value_and_grad` matches the analytic least-squares
gradient at w=0 (parity), and `MlxAdam` drives a tiny regression to
the closed-form solution. Forward + backward + optimizer all on MLX.
`mlx-sys` builds in ~80s with `accelerate` (no Metal/Xcode needed).

**Next (step 006, `mlx-finetune-loop-value-and-grad`):** wire this
into the real MLX LoRA fine-tune loop. When `device("mlx")`, the
fine-tune step builds its loss as a traceable closure over the LoRA
adapter params and uses `loss_and_grads` + `MlxAdam` instead of the
CPU tape (`mlpl-autograd`) + CPU adam (`grad_optim.rs`). Parity-test
the LoRA loss curve vs the CPU path within fp32 tolerance, then
relabel the `MLX LoRA fine-tune` demo hybrid -> true-GPU. The CPU
`backward.rs` 19-fn FAIL is now orthogonal debt (approach A bypasses
it for the MLX path); retire it separately via the sibling-crate split.

### Step 006 slice (2026-05-31): on-device LoRA training kernel proven

`mlpl-mlx-train` now has the full on-device training kernel and proves
the LoRA mechanism end-to-end (it was the linreg spike before):

- `lora_linear(x, w, a, b, scale)` -- the traceable LoRA forward
  `x @ (w + scale*(a@b))` in MLX ops, differentiable w.r.t. the
  adapters.
- `train_steps(params, adam, n, loss_fn)` -- runs the on-device loop
  (forward + `value_and_grad` backward + `MlxAdam` update), returning
  the loss curve. Nothing round-trips to the CPU.

Two gated tests pass on Apple Silicon: `value_and_grad` matches a
finite-difference gradient over BOTH adapters (parity), and `MlxAdam`
drives a frozen-base + rank-1-adapter problem to a collapsed loss.
Crate stays clean (4 modules, sw-checklist 6 passed / 0 failed / 0
warnings). MLX's `value_and_grad` mutates global trace state and is
not thread-safe, so the gradient tests serialize on a crate-local
`MLX_TEST_LOCK` (the parallel eager parity tests in `mlpl-mlx-rt` are
unaffected) -- do not run grad tests in parallel.

**Remaining for the demo (step 007, `mlx-finetune-model-forward`):**
express the FULL demo model forward -- embed (one-hot matmul), causal
attention, rms_norm, linear, cross_entropy -- as `mlx_rs::Array` ops
(the interpreter forward is eager/CPU-materialized and cannot be
traced as-is), assemble it into the `train_steps` loss closure over
the LoRA adapters, wire it into `eval_adam`'s MLX path (read base
params + adapters from the Environment, write adapters back each
step), parity-test the loss curve vs the CPU path
(`lora_mlx_demo_tests.rs`) within fp32 tol, then relabel the demo
hybrid -> true-GPU. This forward reimplementation is the bulk; split
into its own crate (e.g. `mlpl-mlx-forward`) if it grows the module
budget.

### Step 007 slice (2026-05-31): forward primitives in MLX ops

New crate `components/native-rt/crates/mlpl-mlx-forward` (native-rt now
4 crates, at the <=4 budget; sw-checklist 6 passed / 0 failed / 0
warnings) implements the non-attention forward primitives as
`mlx_rs::Array` ops, each parity-tested vs a hand-computed reference:

- `embed(onehot, table)` -- embedding as a one-hot matmul (traceable +
  differentiable w.r.t. the table, unlike an index gather).
- `rms_norm(x, gamma, eps)` -- `x / sqrt(mean(x^2)+eps) * gamma`.
- `cross_entropy(logits, targets_onehot)` -- stable `logsumexp_axis`
  minus the picked logit, mean over rows.

These are forward-VALUE tests (no `value_and_grad`), so they run in
parallel. Remaining for the demo: `causal_attention` (step 008), then
assemble the full forward into the `train_steps` loss closure, wire
into `eval_adam`'s MLX path, parity-test vs the CPU demo, and relabel
the demo true-GPU (step 009).

### Step 008 slice (2026-05-31): causal attention

`mlpl-mlx-forward` gains `attention.rs` (crate stays 4 modules / 0
warnings): `causal_attention(x, wq, wk, wv, wo, mask)` =
`softmax((Q K^T)/sqrt(d_k) + mask) V` then the output projection, plus a
`causal_mask(t)` builder (0 on/below the diagonal, large-negative
above). Single-head (h=1, matching the demo's `causal_attention(d, 1,
_)`); multi-head per-head slabs are a later extension. Parity-tested
vs a hand-computed identity-weight case (T=2): row 0 attends only to
key 0, row 1 mixes by `softmax([0, 1/sqrt2])`.

All forward primitives for the demo model now exist in MLX ops
(embed, rms_norm, causal_attention, linear via `lora_linear`,
cross_entropy). Step 009 assembles them into the `train_steps` loss
closure, wires `eval_adam`'s MLX-LoRA path, parity-tests the loss
curve vs the CPU demo, and relabels the demo true-GPU.

### Step 009 slice (2026-05-31): the full demo model trains on MLX

New component `components/mlx-model/crates/mlpl-mlx-model` (its own
component so native-rt stays at 4 crates; sw-checklist 7 passed / 0
failed / 0 warnings) assembles the demo architecture into one traceable
graph:

- `DemoWeights` -- the frozen base weights (embed table, attention
  Wq/Wk/Wv/Wo, head W + bias, an all-ones gamma, causal mask).
- `demo_forward(weights, adapters, x_onehot, y_onehot)` -- runs
  `embed -> rms_norm -> causal_attention -> (residual) -> rms_norm ->
  lora head -> cross_entropy`. Base weights are captured constants; the
  traced params are the head's single `[A, B]` adapter pair.

A gated test trains the adapters with `MlxAdam` via `train_steps` and
asserts the cross-entropy drops -- gradients flow through the WHOLE
assembled model and the optimizer reduces the loss, all on the GPU.
This is the last technical unknown; what remains is interpreter glue.

### Step 010 progress (2026-05-31): exact CPU-match findings

Wiring requires `demo_forward` to match the CPU `apply_model` EXACTLY
(else the parity test diverges). Reading the CPU primitives corrected
several wrong assumptions (the forward is now fixed accordingly):

- **RMSNorm is gamma-free**, `eps = 1e-8` (`model_apply_compose.rs`):
  `y = x / sqrt(mean(x^2) + eps)`. The `RmsNorm` layer has NO params
  (`params()` returns empty), so the MLX path passes an all-ones gamma.
- **Only the head is LoRA-adapted.** `lora()` rewrites `Linear` ->
  `LinearLora`, but the attention projections live in the separate
  `Attention { wq, wk, wv, wo }` variant, NOT `Linear` -- so they stay
  FROZEN. The demo therefore has ONE adapter pair (`__lora_A_0`,
  `__lora_B_0`) on the head, not one per projection.
- **The head LoRA carries a bias** (`model_apply_lora.rs`):
  `x @ w + (alpha/rank) * (x @ a @ b) + b`.
- **Attention** (h=1, `model_apply_attention.rs`): `scale = 1/sqrt(d_k)`,
  `d_k = d_model/heads`; matches `causal_attention` for the single-head
  demo.

So `mlpl-mlx-model` now also exposes the converters it needs:
`mlpl-mlx-rt::{dense_to_mlx, mlx_to_dense_data, Array}` are now public.

**Remaining (the finish):** add `mlpl-mlx-train` + `mlpl-mlx-model` to
`mlpl-eval`'s `mlx` feature; a new `grad_optim_mlx.rs` that, when
`device("mlx")` and the only non-frozen params are the head adapters
AND the model matches the demo shape (else fall back to CPU): walks the
student `ModelSpec` for the frozen weights + adapter names, extracts X/Y
from the `cross_entropy(apply(model, X), Y)` loss expr, builds one-hots,
runs one `loss_and_grads` + a stateless MLX adam update per step
(persisting m/v in `env.optim_state` like the CPU path), writes the
adapters back. Branch at the top of `eval_adam`. Then assert the
`device("mlx")` loss curve matches the CPU path in `lora_mlx_demo_tests.rs`
within fp32 tol, and relabel the `MLX LoRA fine-tune` demo hybrid ->
true-GPU. (Apple-only glue; see the CUDA-pivot note below.)

## "Measurable learning" -- the success metric

Every training/fine-tuning demo must assert a measurable delta on
HELD-OUT data, not just a falling train loss:

- Language models: validation cross-entropy / perplexity /
  bits-per-char dropping over steps, plus before/after text samples.
- Classification / synthetic tasks: validation accuracy rising.
- The demo prints the metric at step 0 and at the end and a smoke
  test asserts the improvement, so "learning happened" is checkable.

## The ~100MB model: SmolLM2-135M (finale), de-risked first

Target: **SmolLM2-135M** -- a real, instruction-tuned 135M-param
model (~135MB q4 via Ollama; ~270MB bf16 for LoRA training). It is
Ollama-compatible, so the same model can be served for inference AND
be the subject of "ask Ollama for fine-tuning help."

Because loading + fine-tuning a real pretrained transformer is the
biggest unknown, de-risk in order:

1. Showcase measurable learning on the EXISTING MLPL-native MLX LoRA
   path first (it already runs).
2. Then build the SmolLM2 loader + LLaMA-style architecture +
   tokenizer.

### Known gaps for the SmolLM2 finale

- Weight loader: no `safetensors` / `GGUF` reader yet.
- Architecture ops: RoPE, RMSNorm, SwiGLU, grouped-query attention
  are not present (SmolLM2 is a LLaMA-style decoder).
- Tokenizer: SmolLM2's BPE vocab/merges.

## Phased plan

- **Phase 0 -- Ollama settings exposure.** Server-owned default
  host + model (config / flags / `OLLAMA_HOST`); `GET <host>/api/tags`
  -> a `:models ollama` listing + UI picker. (Per-`:ask --model`
  override DEFERRED at user request.)
- **Phase 1 -- Demo capability tiers + gating.** Tag every demo
  with `{ requires_connect, device: cpu|mlx|cuda }`. Separate demo
  sections for `connect`, `mlx`, and (future) `cuda`; `cpu` demos
  (incl. the tiny LoRA fine-tune) run on the public live demo,
  GPU/connect demos render visible-but-not-runnable there. Seed:
  the CPU LoRA fine-tune (live, measurable 6.12->1.34) and the
  contextual `:ask` (connect). The MLX section's training entry is
  HONESTLY labeled "forward on GPU, backward+adam on CPU" until the
  phase below lands.
- **Phase 1b -- Full MLX GPU training loop.** Make the fine-tune
  loop actually run on the GPU: MLX backward + MLX-resident adam
  (extend `mlpl-mlx-rt` and the autograd/optimizer path so
  gradients + moment buffers stay on-device, no per-step CPU
  round-trip). Only after this is the MLX demo a true GPU
  fine-tune. CUDA is the same shape on a Linux peer (later).
- **Phase 1c -- Viz passthrough ("3D everywhere").** The UI -- the
  3D view especially -- should show results regardless of WHERE the
  work ran (local WASM, the MLX peer, a future CUDA peer). Today the
  connect-mode eval response is just `{value, kind}` (a display
  string), so server-evaluated lines emit NO 3D. Make the eval
  response carry viz data so the client emits the 3D event for ANY
  result: (1) `shape` + flat `values` (+ `string_list`) first --
  covers basic tensor/grid/bar sculptures, no new serialization;
  (2) `viz_node` later (attention heatmaps, Sankey) -- requires
  making `VizNode` (mlpl-web-viz-ir) `Serialize`/`Deserialize` and
  extracting it server-side (today that lives in `mlpl-wasm`'s
  `eval_with_values`, not a shared crate). This is the GATE that
  lets `device("mlx")`/`device("cuda")` blocks route to the server
  (GPU) WITHOUT losing 3D -- the real form of the web MLX demo. The
  `mlpl-serve` `mlx` feature (shipped) is the server-side half.
- **Phase 2 -- Agentic tool-using `:ask`.** Server-side
  `/api/chat` loop with `tools`: the model requests context
  (`get_recent_history`, `get_workspace_vars`, `describe_variable`,
  `get_selected_sculpture`, `get_builtin_help`, `get_demo_source`)
  and answers from what it retrieved. Surface the tool-call trace.
- **Phase 3 -- RLM "ask for help" demo.** A demo that fine-tunes on
  the GPU, then issues an agentic `:ask` ("why did the loss
  plateau?"). The model calls the context tools (loss history,
  hyperparameters) and advises -- the visible RLM loop.
- **Phase 4 -- SmolLM2-135M loader + architecture.** safetensors/
  GGUF reader, LLaMA-style ops (RoPE/RMSNorm/SwiGLU/GQA), SmolLM2
  BPE tokenizer; load weights by path and run a forward pass that
  matches a reference.
- **Phase 5 -- The headline demo: download -> measure -> fine-tune
  -> re-measure -> show improvement.** One demo (web in connect mode
  + a CLI twin) that: (1) `fetch_pretrained("smollm2-135m")` with
  size disclosure + cache; (2) measures baseline held-out perplexity
  + shows a baseline generation; (3) LoRA fine-tunes on a small
  target corpus on the GPU; (4) re-measures perplexity + shows a
  post-fine-tune generation; (5) asserts + displays the improvement.
  Then an agentic `:ask` advises on the fine-tune from the real
  metrics, and the result can be served via Ollama.

## 3D dimension orientation (cross-cutting, normative)

All 3D sculptures must follow `docs/3d-orientation.md`: longest
dimension recedes toward the mountains (-Z), 2nd-longest rises to
the sky (+Y), 3rd spreads left/right (X); stacked maps (attention
heads, conv channels) lay out as a row of maps receding along -Z.
Phase 1c part 1 fixed the per-head attention overlay
(`multiHeadStrip`); rank-2 matrices + conv stacks remain to align.
This rule keeps getting re-broken -- renderers cite the doc inline.

## References

- `docs/3d-orientation.md` -- normative axis-mapping rule above.
- `docs/SmolLM2-demo-plan.md` -- the pre-existing SmolLM2 135M
  LoRA-on-MLX demo plan. Phases 4-5 here build on it; note its
  decision that base weights + adapters live OUTSIDE this language
  repo (in `softwarewrighter/efficient-llm`), which refines the
  "demo-initiated download" policy above: the fetch caches into an
  external location, not the MLPL repo tree.
- `docs/agentic-ask-plan.md` -- the agentic `:ask` detail.
- `docs/using-llm-tool.md`, `contracts/eval-contract/llm-call.md`.
- `components/models-write/crates/mlpl-models-tune/` -- `lora()`.
- `components/native-rt/crates/mlpl-mlx-rt/`,
  `components/serve/crates/mlpl-mlx-serve/` -- MLX path.
- Recursive Language Models (external) -- the agentic inspiration.
