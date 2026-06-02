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
- **Phase 1 -- Local GPU demo group + live-demo gating.** A "Local
  GPU" demo category. Verify `lora_finetune_mlx` shows measurable
  learning (held-out metric + smoke assertion). Add a
  `requires_connect` flag so connect/GPU-only demos render
  visible-but-not-runnable on the public GitHub Pages demo.
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

## References

- `docs/agentic-ask-plan.md` -- the agentic `:ask` detail.
- `docs/using-llm-tool.md`, `contracts/eval-contract/llm-call.md`.
- `components/models-write/crates/mlpl-models-tune/` -- `lora()`.
- `components/native-rt/crates/mlpl-mlx-rt/`,
  `components/serve/crates/mlpl-mlx-serve/` -- MLX path.
- Recursive Language Models (external) -- the agentic inspiration.
