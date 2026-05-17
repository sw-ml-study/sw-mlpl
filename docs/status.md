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
| R1 | MLX as a service (`services/mlpl-mlx-serve`; orchestrator `--peer mlx=<url>` routing for `device("mlx") { ... }`; opaque `DeviceTensor` handles; strict CPU faults until `to_device("cpu", x)` materialization; in-process MLX fallback retained) | v0.18.0 | [x] |
| 23 | Typed ML values + typed traces (Tier A vocabulary: Logit / Probability / LogProbability / Loss{kind} / Gradient{wrt} / Weight{layer,name} / Bias{layer} / Activation{layer,kind} / LearningRate / Labels{num_classes} / AttentionMap; auto-tagging from softmax/sigmoid/cross_entropy/grad/cosine_schedule/linear_warmup/attention_weights producers + linear/embed/attention param creation + apply structural-tail rule; predicate consumers with `EvalError::TypeMismatch` four-part tutoring hints; tag propagation through arith/transpose/reshape/reductions/negation; new `:tags` / `:untag` and typed `:describe` / `:vars`; typed trace JSON events; "Typed ML Values" web REPL lesson; gradual-typing additivity preserved; performance explicitly last in the goal ranking) | v0.19.0 | [x] |
| 21.5 | Multi-client UI follow-up (SSE streaming eval at `POST /v1/sessions/{id}/eval_stream`, cancellation at `POST /v1/sessions/{id}/cancel` + `EvalError::Cancelled` + per-session interrupt token, viz storage at `POST /v1/viz` + `GET /v1/viz/{sha}.{ext}` with five-format detect, web REPL connect mode via `Evaluator` trait + `?server=<url>` query string + `--cors-allow`, session persistence via `--persist <dir>` + `persist_version: 1` JSON-on-disk, session reattach via `--session <id> --token <tok>` + `GET /v1/sessions/{id}`, f32 + u8 dtypes on the MLX peer wire as a tagged 3-variant union; LLM proxy / WebSocket / model+optimizer persistence deferred to follow-ups) -- see `docs/milestone-multi-client-followup.md` | v0.20.0 | [x] |

## Planned

Intended sequence: **29 -> (dev host move to
Linux) -> R2 -> R3 -> 18**. Saga 21.5 shipped in
v0.20.0 and unblocks the browser REPL driving the
MLX peer, which is what Saga 29 (Vision
Transformer) needs for the multi-head thorough
demo. R2 / R3 follow the dev-host move to Linux.
Saga 17 was superseded by the services refactor
proposed in `docs/refactor-services.md`; R1
shipped in v0.18.0 and R2 / R3 replace the
remaining in-process CUDA / distributed portions.
The typed-values follow-ups (24-28) can interleave
once a track lead picks them up.

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 29 | Vision Transformer track (`load_images`, `load_preloaded("pets_tiny")`, `fetch_dataset("oxford_iiit_pet")`, `patchify`, `concat`, `gelu`, `layer_norm`, multi-head attention on the tape; single-head quick demo + multi-head thorough demo on MLX) -- see `docs/milestone-vit.md` and `docs/ViT-demo-plan.md` | v0.21.0 | [ ] | 21.5 (for thorough demo), R1 |
| R2 | CUDA-as-a-service (`mlpl-cuda-serve`; same shape as R1; replaces the originally-planned in-process Saga 17) | tbd | [ ] | R1, dev host move |
| R3 | Distributed primitives + LAN auto-discovery (`run model on nodes[...]`, mDNS peer discovery, peer-to-peer tensor migration) | tbd | [ ] | R1, R2 |
| 17 | CUDA backend and distributed execution -- **SUPERSEDED** by R1 / R2 / R3; see `docs/refactor-services.md` | -- | [-] superseded | -- |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | tbd | [ ] | 15 |
| 24 | First-class Distributions (`Categorical`, `Gaussian`, `Mixture`; `sample` / `log_prob` / `entropy` / `kl_divergence`; reparam gradients for Gaussian; VAE / policy-gradient / mixture-density demos) -- see `docs/milestone-distributions.md` | tbd | [ ] | 23 |
| 25 | Inspectable ComputationGraph (`Value::Graph`, `compute_graph(loss)`, static `svg(g, "compute_graph")`, `animate(g)` carousel, `jacobian` / `hessian`) -- see `docs/milestone-compute-graph.md` | tbd | [ ] | 23 |
| 26 | Annotation syntax + tutoring errors (extend Saga 11.5 colon-annotation to type names; assignment-time tag predicate; assignment-site tutoring catalog; typed builtin signatures in `:describe`) -- see `docs/milestone-typed-annotations.md` | tbd | [ ] | 23, 11.5 |
| 27 | Typed Layer roles + walked `:describe mdl` (Tier B: `LayerRole`, first-apply shape pinning, walked typed-tree `:describe`, `:hidden mdl k`, typed Optimizer / Schedule / Dataset roles) -- see `docs/milestone-typed-layers.md` | tbd | [ ] | 23, 11 |
| 28 | User-defined tags (`define_tag` registry, `tag(x, "...")` attachment, curated invariant vocabulary, trace + describe integration, promotion-to-curated path) -- see `docs/milestone-user-tags.md` | tbd | [ ] | 23, 26 |
| -- | QLoRA / 4-bit quantization (deferred follow-up from Saga 15) | tbd | [ ] | 15 |
| -- | UMAP reducer (deferred follow-up from Saga 16; overlaps with t-SNE) | tbd | [ ] | 16 |
| -- | RAG pipeline over a local LLM inference path (deferred follow-up from Saga 16) | tbd | [ ] | 16, 19 |
| -- | Interactive 3-D scatter (rotation/zoom) + MLX dispatch for t-SNE (deferred follow-ups from Saga 16) | tbd | [ ] | 16 |
| -- | Static type checks on the `mlpl!` / `mlpl build` lower path (deferred follow-up to Sagas 23-26; lifts annotation predicates to lower time) | tbd | [ ] | 23, 26 |
| -- | Server-side LLM proxy with allow-list (split from Saga 21.5 pending its own security review) | tbd | [ ] | 21.5, 19 |

## Next saga to start

**Saga 29 -- Vision Transformer track.** Saga 21.5
shipped in v0.20.0 with no remaining steps: the
browser REPL now drives the MLX peer over SSE +
the viz storage endpoint, sessions persist across
restarts, and the wire format takes f32 / u8 in
addition to f64. The next saga lands the image-
tensor primitives (`load_images`,
`load_preloaded("pets_tiny")`,
`fetch_dataset("oxford_iiit_pet")`, `patchify`,
`concat`, `gelu`, `layer_norm`, multi-head
attention on the tape) and the four-demo ladder
in `docs/ViT-demo-plan.md` /
`docs/milestone-vit.md`. Steps 001-005 (Tier 1
builtins + single-head quick demo) can run on the
Mac dev host today; step 008+ (multi-head,
thorough, browser-against-MLX) was the path
Saga 21.5 was opening up. The Phase 4.5 gallery /
predict_batch / BYO image UX features in
`docs/ViT-demo-plan.md` Demo 5 + Demo 6 are part
of the same saga.

Saga R2 -- CUDA-as-a-service -- still follows the
dev-host move to Linux, after 29. R3 distributed
primitives + mDNS auto-discovery layer on top of
the two concrete service backends.
