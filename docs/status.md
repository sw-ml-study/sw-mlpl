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
| 21 | CLI server + multi-client UI MVP (`crates/mlpl-serve` REST skeleton: sessions / eval / inspect / health; `mlpl-repl --connect <url>` thin client; CLI viz cache `MLPL_CACHE_DIR`; constant-time auth; LLM proxy / SSE / cancellation / persistence / web-rerouting / Emacs / TUI / desktop-GUI all deferred to follow-up sagas) | v0.17.0 | [x] |

## Planned

Intended sequence: **(dev host move to Linux) -> 17 -> 18**.
Post-MVP follow-ups to Saga 21 (LLM proxy, SSE, cancellation,
persistence, web UI re-routing, ratatui / Emacs / desktop GUI
clients) fold into a follow-up CLI-server saga after the MVP
server contract proves stable in real use; not gated on the
Linux move.

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 17 | CUDA backend and distributed execution (requires Linux + NVIDIA host) | v0.18.0 | [ ] | 14, dev host move |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | v0.19.0 | [ ] | 15 |
| -- | QLoRA / 4-bit quantization (deferred follow-up from Saga 15) | tbd | [ ] | 15 |
| -- | UMAP reducer (deferred follow-up from Saga 16; overlaps with t-SNE) | tbd | [ ] | 16 |
| -- | RAG pipeline over a local LLM inference path (deferred follow-up from Saga 16) | tbd | [ ] | 16, 19 |
| -- | Interactive 3-D scatter (rotation/zoom) + MLX dispatch for t-SNE (deferred follow-ups from Saga 16) | tbd | [ ] | 16 |

## Next saga to start

**Dev host move to Linux + NVIDIA, then Saga 17 (CUDA +
distributed execution).** Saga 21 MVP (v0.17.0) just
shipped: `crates/mlpl-serve` skeleton, the
`mlpl-repl --connect <url>` thin client, and the CLI
visualization cache (`MLPL_CACHE_DIR`,
content-addressed SHA-prefix paths). The
`mlpl-serve` running natively on the Apple-Silicon
host with `--features mlx` is the bridge that keeps
MLX-accelerated training reachable from any client
once dev moves off Apple Silicon. Saga 17 then
adds CUDA kernels (paralleling Saga 14's MLX
fused ops) plus distributed primitives
(`run model on nodes[...]`, device placement,
basic data-parallel training). Homelab LAN
training demo as the headline. Post-MVP follow-ups
to Saga 21 (server-side LLM proxy with allow-list,
visualization storage URLs, Server-Sent-Events
streaming, cancellation, persistence, web UI
re-routing, ratatui / Emacs / desktop GUI clients)
slot in as a follow-up CLI-server saga whenever
the MVP server contract proves stable in real use
-- not gated on the Linux move.
