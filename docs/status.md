# MLPL Saga Status

Snapshot of every saga, completed and planned. Forward-looking
detail and rationale live in `docs/plan.md`; this file is the
one-line-per-saga scoreboard.

Legend: [x] complete  [~] in progress  [ ] planned  [-] deferred

## Completed

| # | Saga | Version | Status |
|---|------|---------|--------|
| -1 | Repo scaffolding | -- | [x] |
| 0 | Foundation and contracts | -- | [x] |
| 1 | Dense tensor substrate v1 | -- | [x] |
| 2 | Parser and evaluator foundation | -- | [x] |
| 3 | CLI and REPL v1 | v0.1 | [x] |
| 4 | Structured trace v1 (JSON export) | v0.1 | [x] |
| 5 | Visual web viewer v1 | -- | [-] deferred post-MVP |
| 6 | ML foundations (matmul, activations, logistic regression) | v0.2 | [x] |
| 7 | SVG visualization v1 (`mlpl-viz`, inline SVG in REPL) | v0.3 | [x] |
| 8 | ML demos (k-means, PCA, softmax, tiny MLP, attention) | v0.4 | [x] |
| 9 | Autograd v1 (reverse-mode tape, `grad` built-in) | v0.5 | [x] |
| 10 | Optimizers + training loop (Adam, schedules, moons/circles, `train { }`) | v0.6 | [x] |
| 11 | Model DSL (`chain`, `residual`, `attention`, `norm`, differentiable `apply`) | v0.7 | [x] |
| 11.5 | Named axes and shape introspection (`label`, annotation syntax, ShapeMismatch, trace labels) | v0.7.5 | [x] |
| -- | Compile-to-Rust (`mlpl!` macro, `mlpl build`, parity harness, 9x speedup) | v0.8.0 | [x] |
| 12 | Tokenizers, datasets, experiment tracking (`load`, `shuffle`/`batch`/`split`, `for`, byte-level BPE, `experiment`/`:experiments`/`compare`) | v0.9.0 | [x] |
| 13 | Tiny language model end-to-end (`embed`, `sinusoidal_encoding`, `causal_attention`, `cross_entropy`, `sample`/`top_k`, `attention_weights`, `tiny_lm`/`tiny_lm_generate`) | v0.10.0 | [x] |
| 14 | MLX backend (`mlpl-mlx` runtime, `device("...") { }` scoped form, `to_device`, autograd + optimizers + `train { }` on MLX, `tiny_lm_mlx` demo) | v0.11.0 | [x] |
| 20 | Perturbation research demos / Neural Thickets (`clone_model`, `perturb_params`, `argtop_k`, `scatter`, `neural_thicket` CPU + MLX demos, specialization heatmap) | v0.12.0 | [x] |
| 15 | LoRA fine-tuning (`freeze`, `unfreeze`, `lora`, `LinearLora` variant, `lora_finetune` CPU + MLX demos, CPU-MLX parity within fp32; QLoRA / quantization deferred) | v0.13.0 | [x] |
| 16 | Embedding visualization (`pairwise_sqdist`, `knn`, `tsne`, `svg(..., "scatter3d")`, `embedding_viz` demo; UMAP / RAG / MLX-for-tsne deferred) | v0.14.0 | [x] |
| 16.5 | Embedding-viz polish (`pca(X, k)` + `embed_table(model)`; demo / docs / tutorial updates; UMAP / interactive 3-D / MLX-for-tsne stay deferred) | v0.14.1 | [x] |
| 22 | Feasibility checking + resource estimation (`estimate_train`, `calibrate_device`, `estimate_hypothetical`, `feasible`; SmolLM / Llama / Qwen what-if table; design deviation: direct `estimate_hypothetical` instead of `hypothetical_model` -> ModelSpec) | v0.15.0 | [x] |
| 19 | LLM-as-tool REST integration (`llm_call(url, prompt, model) -> string`; `:ask` migrated onto shared HTTP path; CLI-only pending Saga 21 proxy; streaming/tools/chat-threading/batching/auth/web all deferred) | v0.16.0 | [x] |

## Planned

Intended sequence: **21 -> (dev host move to Linux) -> 17 -> 18**.
Saga 21 (CLI server) is prioritized **before** the Linux move because once
dev is off Apple Silicon the browser live demo loses local MLX; the
server-side `mlpl-serve` route keeps it usable.

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 21 | CLI server + multi-client UI (REST, proxy, CLI/web/Emacs clients) | v0.17.0 | [ ] | 13 |
| 17 | CUDA backend and distributed execution (requires Linux + NVIDIA host) | v0.18.0 | [ ] | 14, dev host move |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | v0.19.0 | [ ] | 15 |
| -- | QLoRA / 4-bit quantization (deferred follow-up from Saga 15) | tbd | [ ] | 15 |
| -- | UMAP reducer (deferred follow-up from Saga 16; overlaps with t-SNE) | tbd | [ ] | 16 |
| -- | RAG pipeline over a local LLM inference path (deferred follow-up from Saga 16) | tbd | [ ] | 16, 19 |
| -- | Interactive 3-D scatter (rotation/zoom) + MLX dispatch for t-SNE (deferred follow-ups from Saga 16) | tbd | [ ] | 16 |

## Next saga to start

**Saga 21 -- CLI server + multi-client UI.** Saga 19 (v0.16.0)
just shipped `llm_call(url, prompt, model)` as a CLI-only
language-level builtin; the browser path was deferred to
Saga 21 because localhost Ollama is unreachable from the
browser without CORS allow-listing + a server-side proxy.
Saga 21 builds that proxy. New `crates/mlpl-serve` binary
exposing a REST + WebSocket surface over a long-running
MLPL interpreter; one server, many clients (web UI wired
to call origin -- unblocks `:ask` / `llm_call` from the
browser via a server-side allow-listed Ollama proxy --
plus `mlpl-repl --connect <host>`, eventual ratatui TUI,
Emacs client). Session isolation per bearer token;
`--bind 0.0.0.0` requires `--auth required`; proxy
allow-list gates which LLM providers are reachable and
server-side env vars hold their keys. Ships with the CLI
visualization strategy (auto-write SVG to cache dir + print
path) so the terminal REPL stops dumping raw `<svg>` XML.
Prioritized before the Linux move because once dev goes off
Apple Silicon the browser live demo loses local MLX; the
server-side `mlpl-serve` keeps it usable. See
`docs/configurations.md` for the architecture diagram + REST
API sketch + security posture.
