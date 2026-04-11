# Are We Driven Yet?

**Purpose.** Track MLPL's progress against the two forcing-function
documents that drive its long-range design:

- `docs/paper-driven-development.txt` -- a curated backlog of ~20
  recent ML papers (Dec 2025 - Apr 2026) and the MLPL primitives
  each one would exercise.
- `docs/ai-agent-driven-development.txt` -- a wishlist from the
  perspective of an LLM/agent that wants to use MLPL as its
  reasoning and execution substrate.

Both documents are speculative discussions, not plans of record.
This file is the accounting sheet that turns them into actionable
status: for every capability they call out, does MLPL already have
it, is it planned in `docs/plan.md`, or does it still need to be
shaped into a saga?

## Legend

| Symbol | Meaning |
|--------|---------|
| `HAVE` | Shipped, exercised by demos/tests today |
| `PART` | Partially implemented, needs deepening to satisfy the source doc |
| `PLAN` | Already on the Saga 12-19 roadmap in `docs/plan.md` |
| `CONS` | Not planned; needs design + a saga shape before it can move |

**Source** column: `P` = from `paper-driven-development.txt`, `A` =
from `ai-agent-driven-development.txt`, `P+A` = both.

---

## 1. Core language and array semantics

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| Dense tensors with row-major storage | A | HAVE | `mlpl-array` |
| Scalar broadcasting | A | HAVE | Scalar op array across all primitives |
| Element-wise arithmetic and reductions | A | HAVE | `reduce_add`/`reduce_mul` with axis arg |
| Matrix multiplication, transpose | A | HAVE | `matmul`, `transpose`, `dot` |
| Shape introspection (`shape`, `rank`) | A | HAVE | Built-ins |
| Repeat loop and `train N { }` sugar | P+A | HAVE | Saga 10 |
| Small core language (few, composable primitives) | A | HAVE | Intentional design constraint |
| Canonical formatting of MLPL source | A | CONS | No MLPL-level formatter exists; Rust side uses rustfmt. Needed for LLM roundtripping |
| REPL with incremental execution | A | HAVE | `mlpl-repl` + web REPL |
| Named axes on tensors (`x[batch, time, dim]`) | A | PLAN | Saga 11.5 (`docs/milestone-named-axes.md`) |
| Shape-checked / dependent-shape types (`Tensor[B, T, D]`) | A | PART | Saga 11.5 tracks labels as metadata with runtime validation, stopping short of full dependent types |
| Dual syntax: terse APL mode vs explicit agent-friendly mode | A | CONS | Open design question |
| Code-as-data: `parse`, `transform`, `eval` primitives | A | CONS | Needed for meta-programming, self-modification |
| Deterministic execution mode (`with deterministic { }`) | A | CONS | Seed control needs a scoped construct |
| Deterministic serialization for programs + models | A | CONS | Needed for repeatable agent loops |
| Notebook mode with persisted cell state | A | CONS | Web REPL has session state but no cell model |

## 2. Autograd, optimizers, training

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| Reverse-mode autograd (`grad(expr, wrt)`) | A | HAVE | Saga 9 |
| `param[shape]` trainable-leaf constructor | A | HAVE | Saga 9 |
| `momentum_sgd`, `adam` with persistent state | P+A | HAVE | Saga 10 |
| `cosine_schedule`, `linear_warmup` helpers | P | HAVE | Saga 10 |
| Model DSL (`linear`/`chain`/activations) | P+A | HAVE | Saga 11, training through `apply()` wired 2026-04-09 |
| Model DSL: `residual` / `rms_norm` / `attention` forward | P | HAVE | Saga 11 steps 004-005 |
| Model DSL: differentiable `apply()` through residual/norm/attention | P | HAVE | Tape-lowered for linear/chain/activations/residual/rms_norm/single-head attention in Saga 11 step 007 (fb90fb4). Multi-head (`heads>1`) still forward-only pending per-head slicing |
| `jacobian f`, `vmap f` | A | CONS | Not on the Saga 11 roadmap; Jax-style transforms would be a Saga 11.5 item |
| Named-parameter trees and parameter addressing | P | PART | `ModelSpec::params()` returns a flat list; no path-addressed subtree ops |
| Hyperparameter sweep DSL (adapter/rank/lr grid) | P | PLAN | Saga 12 ("experiment tracking") |
| Experiment registry + comparison plots | P | PLAN | Saga 12 |
| Hessian / eigenvalue / curvature probes | P | CONS | Paper #5 (Learning Rate Matters) asks for this; no plan |
| LoRA / QLoRA adapter declarations | P | PLAN | Saga 15 |
| 4-bit / low-precision quantization | P | PLAN | Saga 15 |
| Self-distillation pipeline primitives (sample -> filter -> finetune) | P | CONS | Papers #17-18; not on roadmap |
| RLVR / token-level credit assignment | P | CONS | Paper #17; not on roadmap |

## 3. Datasets and data pipelines

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| Synthetic datasets (`moons`, `blobs`, `circles`, `randn`) | P+A | HAVE | Saga 8/10 |
| `load.csv`, generic file loaders | A | PLAN | Saga 12 |
| Streaming / lazy dataset ops (`shuffle`, `batch`, `map`) | A | PLAN | Saga 12 |
| Byte-level BPE tokenizer | P | PLAN | Saga 12 |
| Long-context / 100M-token loader | P | CONS | Paper #13 (MSA); needs backend work |
| External memory store (key-value table, writeback, sparse addressing) | P | PLAN | Saga 18 ("engram memory") covers the idea; no concrete API |

## 4. Visualization, tracing, demos

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| Inline SVG visualization (scatter/line/bar/heatmap) | P+A | HAVE | Saga 7 |
| Analysis helpers (`hist`, `scatter_labeled`, `loss_curve`, `confusion_matrix`, `boundary_2d`) | P+A | HAVE | Saga 8 |
| Execution trace with JSON export | A | HAVE | `mlpl-trace` |
| Yew/WASM web REPL | P+A | HAVE | Saga 5 |
| Tutorial mode in web REPL | A | HAVE | 20 lessons including "Model Composition (the Model DSL)" added in Saga 11 |
| Numeric output summarization (collapsible stats) | A | HAVE | Added 2026-04-09 |
| Embedding viz: PCA / t-SNE / UMAP projections | P+A | PLAN | Saga 16 |
| 3D scatter via SVG projection, nearest-neighbor links | P | PLAN | Saga 16 |
| Topology / block-diagram view of model graphs | P | CONS | Paper #2 (mHC) wants it; needs a `ModelSpec` -> SVG pass |
| Debate / "society of thought" transcript viewer | P | CONS | Paper #19; not on roadmap |
| Skill graph + reuse frequency chart | P | CONS | Paper #10; not on roadmap |
| Kernel performance leaderboard + diff view | P | CONS | Paper #12 (CUDA Agent); not on roadmap |
| PaperBanana-style slide generation from a run | P | CONS | Artifact generator; not on roadmap |

## 5. Backends and runtime integrations

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| CPU dense kernels (current default) | A | HAVE | `mlpl-array` |
| Explicit device placement syntax (`x \|> cuda`, `with mlx { }`) | A | CONS | Required by both source docs; no construct chosen yet |
| MLX backend | P+A | PLAN | Saga 14 |
| Lazy execution graph + kernel fusion | P | PLAN | Saga 14 |
| CUDA backend | P+A | PLAN | Saga 17 |
| Distributed execution (`run model on nodes[...]`) | P | PLAN | Saga 17 |
| Ollama / llama.cpp unified model interface | P+A | PLAN | Saga 19 (REST client built-ins) |
| Streaming `generate` with incremental tokens | A | CONS | Saga 19 covers REST; streaming as a language construct is not pinned down |
| Profiler ingestion / FLOPs estimation | P+A | CONS | Paper #12 wants it as a reward signal; no plan |

## 6. Agent-facing ergonomics

| Capability | Source | Status | Notes |
|------------|--------|--------|-------|
| Introspectable execution graph (`trace f`) | A | HAVE | `mlpl-trace` JSON export |
| Structured errors (kind + expected/actual fields) | A | PLAN | Saga 11.5 phase 4 introduces a structured `ShapeMismatch` error variant and a JSON error channel |
| Built-in testing primitives (`assert`, `approx_equal`, `fuzz`) | A | CONS | Rust tests exist, MLPL-level asserts do not |
| Sandbox execution with time/memory/GPU limits (`run_safe`) | A | CONS | Needed for agent self-correction loops; no plan |
| Cost / performance introspection (`profile f x`) | A | CONS | Paper #12 also benefits; no plan |
| `:describe <name>` / `:vars` / `:fns` REPL introspection | A | HAVE | Shipped in Saga 11 (99f7c8a, 623652d): `:vars`, `:models`, `:fns` user / `:builtins`, `:wsid`, `:describe`, `:help <topic>` in both `mlpl-repl` and `mlpl-web` |
| `explain f` structured-explanation primitive | A | CONS | Speculative but high-leverage for agent tooling |
| `repair f given error` primitive | A | CONS | Speculative; depends on structured errors + code-as-data |
| `optimize f for gpu` primitive | A | CONS | Speculative; depends on device placement + profile |
| Tool schemas and tool-call execution | P | PLAN | Saga 18 ("ICL/ICRL") + Saga 19 |
| Trajectory / episode / rollout / reward records | P | PLAN | Saga 18 |
| Controller/worker multi-model orchestration | P | PLAN | Saga 18 ("multi-model orchestration") |
| Replay buffers and curriculum control | P | PLAN | Saga 18 |

## 7. Paper-specific primitives (the 20-paper backlog)

Papers from `docs/paper-driven-development.txt` ranked backlog. Each
row is "what MLPL would need to make this paper a one-screen demo."

| # | Paper | MLPL primitive asked for | Status |
|---|-------|--------------------------|--------|
| 1 | Recursive Language Models | Recursive task/decompose/reduce constructs; scheduler for nested subcalls with budget + memoization | CONS |
| 2 | mHC: Manifold-Constrained Hyper-Connections | First-class model-graph syntax for residual-stream variants and constrained projection ops | PART (ModelSpec exists, constrained projections do not) |
| 3 | Conditional Memory via Scalable Lookup | Memory tables, lookup, writeback, sparse addressing as first-class ops | PLAN (Saga 18) |
| 4 | In-Context RL for Tool Use | `episode`, `rollout`, `reward`, `tool`, fewshot curriculum syntax | PLAN (Saga 18) |
| 5 | Learning Rate Matters (LoRA tuning) | Sweep DSL (`adapter`, `rank`, `lr`, `scheduler`, `seed`, `judge`) | PLAN (Saga 12 + 15) |
| 6 | RelayLLM | Controller/worker decoding, token-level help requests, cooperative generation | PLAN (Saga 18 + 19) |
| 7 | CM2 Checklist Rewards | Declarative checklist reward specs with evidence anchors | CONS |
| 8 | PEARL planner/executor | Separate `plan { }` and `execute { }` stages with distinct rewards | CONS |
| 9 | Tool-R0 self-evolving agents | Self-play / dual-agent curriculum loops | CONS |
| 10 | SkillRL | `skill` as a reusable packaged abstraction (code + prompt + tests + permissions) | CONS |
| 11 | Agent World Model | Environment DSL for executable synthetic worlds and tools | CONS |
| 12 | CUDA Agent | Kernel blocks, verification hooks, profiler-aware reward terms | PART-PLAN (Saga 17 provides CUDA; codegen loop is unplanned) |
| 13 | MSA (100M-token attention) | Long-context memory/attention operators; memory-interleaving | CONS |
| 14 | Self-Indexing KVCache | Cache-policy syntax, programmable inference-time memory layout | CONS |
| 15 | S3-Attention | `retrieve` from attention primitives blurring retrieval + inference | CONS |
| 16 | iGRPO self-feedback | `draft`, `select_best`, `refine` syntax for two-stage loops | CONS |
| 17 | Self-Distilled RLVR | Token-level credit assignment and distilled-target generation | CONS |
| 18 | Self-distillation for code | `sample -> filter -> finetune` pipeline primitives | CONS |
| 19 | Societies of Thought | `voices`, `debate`, `reconcile`, `critic`, `judge` multi-perspective syntax | CONS |
| 20 | Attention Residuals | Residual-policy syntax; skip/accumulate as first-class choice | PART (ModelSpec has `residual`; no policy selection) |

---

## Aggregate scoreboard

| Bucket | Count | What it means |
|--------|-------|---------------|
| HAVE   | ~28   | Shipped, exercised by demos/tests |
| PART   | ~5    | Partially there; needs deepening (notably multi-head attention tape lowering, named-parameter tree addressing, mHC architecture graph) |
| PLAN   | ~17   | On the Sagas 11.5 / 12-19 roadmap |
| CONS   | ~30   | Not on the roadmap; each is a candidate for a new saga or an inserted step |

## What this tells us about priorities

Historical note (2026-04-09): the three clusters originally listed
here were (1) named axes + shape types, (2) structured errors +
trace-driven self-correction, and (3) differentiable `apply`
through residual/rms_norm/attention. Cluster 3 shipped in Saga 11
steps 006 and 007. The REPL-introspection subset of cluster 2
(`:vars`, `:models`, `:fns`, `:wsid`, `:describe`) also shipped in
Saga 11 ahead of schedule. Clusters 1 and the remaining structured
errors half of cluster 2 landed as **Saga 11.5 -- Named axes and
shape introspection** (`docs/milestone-named-axes.md`), inserted
between Saga 11 and Saga 12 and now NEXT.

With Saga 11 shipped as v0.7.0-modeldsl, the remaining uncovered
region is the rest of the **agent-as-user** half: device
placement, profiling, sandboxing, deterministic mode, code-as-data,
testing, notebook mode, canonical formatting. None of these fit
naturally into Sagas 12-19 as currently scoped. A dedicated
"Saga 11.6: Agent ergonomics" milestone would be a reasonable home
for the smaller ones; the bigger ones (code-as-data, notebook) are
probably Saga-sized on their own.

On the **paper-driven** side, the 20-paper backlog has four
genuine design questions that no current saga covers:

- Recursive task orchestration (paper #1)
- Skill packaging (paper #10)
- Environment DSL (paper #11)
- Checklist / rubric-composed rewards (papers #7, #8)

These are not blockers for any existing saga but they collectively
justify reordering Saga 18 to land sooner, or splitting it into two
sagas along the "primitives first, orchestration second" seam.

## How to use this file

- Before opening a new saga, check the CONS table and see if there
  is a capability that blocks it.
- Before adding a paper demo, confirm the row in section 7 has
  moved out of CONS, or accept that the demo will inline the
  missing primitive.
- Update this file whenever a capability crosses buckets. Keep it
  honest -- the point is to know which side (papers or agents) is
  pulling MLPL forward and which side is falling behind.
