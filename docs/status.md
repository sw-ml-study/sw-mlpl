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

Intended sequence: **R1 -> (dev host move to Linux)
-> R2 -> R3 -> 18**. Saga 17 was superseded by the
services refactor proposed in
`docs/refactor-services.md`; R1 / R2 / R3 replace
it. Post-MVP follow-ups to Saga 21 (LLM proxy, SSE,
cancellation, persistence, web UI re-routing,
ratatui / Emacs / desktop GUI clients) fold into a
follow-up CLI-server saga after the MVP server
contract proves stable in real use; not gated on
the Linux move.

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| R1 | mlpl-mlx as a service (`mlpl-mlx-serve` binary; orchestrator routes `device("mlx") { ... }` blocks to peers; in-process MLX feature stays as fallback; separate `target/` per service workspace -- disk pressure relief) | tbd | [ ] | 21 |
| R2 | CUDA-as-a-service (`mlpl-cuda-serve`; same shape as R1; replaces the originally-planned in-process Saga 17) | tbd | [ ] | R1, dev host move |
| R3 | Distributed primitives + LAN auto-discovery (`run model on nodes[...]`, mDNS peer discovery, peer-to-peer tensor migration) | tbd | [ ] | R1, R2 |
| 17 | CUDA backend and distributed execution -- **SUPERSEDED** by R1 / R2 / R3; see `docs/refactor-services.md` | -- | [-] superseded | -- |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | tbd | [ ] | 15 |
| -- | QLoRA / 4-bit quantization (deferred follow-up from Saga 15) | tbd | [ ] | 15 |
| -- | UMAP reducer (deferred follow-up from Saga 16; overlaps with t-SNE) | tbd | [ ] | 16 |
| -- | RAG pipeline over a local LLM inference path (deferred follow-up from Saga 16) | tbd | [ ] | 16, 19 |
| -- | Interactive 3-D scatter (rotation/zoom) + MLX dispatch for t-SNE (deferred follow-ups from Saga 16) | tbd | [ ] | 16 |

## Next saga to start

**Saga R1 -- `mlpl-mlx` as a service (services
refactor).** Saga 21 MVP (v0.17.0) just shipped:
`crates/mlpl-serve` skeleton, the `mlpl-repl
--connect <url>` thin client, and the CLI
visualization cache (`MLPL_CACHE_DIR`,
content-addressed SHA-prefix paths). The next
high-leverage move is the services refactor
proposed in `docs/refactor-services.md`: promote
`mlpl-mlx` from an in-process Cargo feature to a
standalone service binary (`mlpl-mlx-serve`) that
the orchestrator dispatches `device("mlx") { ... }`
blocks to. Three wins at once: (1) cross-host MLX
from a Linux dev host (orchestrator on Linux, MLX
peer on the Mac); (2) hardware-coupling escape so
R2 (CUDA-as-a-service, replacing the deferred
in-process Saga 17) can land cleanly; (3) disk-
pressure relief because each service workspace
has its own `target/`. R1 keeps the in-process
MLX feature as a fallback so single-host MLX
users don't regress; R2 follows once R1's protocol
is settled; R3 layers distributed primitives +
mDNS auto-discovery on top.

Post-MVP follow-ups to Saga 21 (server-side LLM
proxy with allow-list, visualization storage
URLs, Server-Sent-Events streaming, cancellation,
persistence, web UI re-routing, ratatui / Emacs /
desktop GUI clients) slot in as a follow-up
CLI-server saga whenever the MVP server contract
proves stable in real use -- not gated on R1 / R2
/ R3 or the Linux move.
