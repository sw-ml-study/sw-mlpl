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

## Planned

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 12 | Tokenizers, datasets, experiment tracking | v0.8 | [ ] NEXT | 11.5 |
| 13 | Tiny language model end-to-end (CPU) | v0.9 | [ ] | 12 |
| 14 | MLX backend (Apple Silicon, lazy graph, fusion) | v0.10 | [ ] | 13 |
| 15 | LoRA / QLoRA / quantization | v0.11 | [ ] | 13 |
| 16 | Embedding visualization and RAG | v0.12 | [ ] | 13 |
| 17 | CUDA backend and distributed execution | v0.13 | [ ] | 14 |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | v0.14 | [ ] | 15 |
| 19 | LLM-as-tool REST integration | v0.15 | [ ] | 12 |

## Next saga to start

**Saga 12 -- Tokenizers, datasets, and experiment tracking.** With
Saga 11.5 shipped in v0.7.5 (labeled shape metadata,
`x : [batch, time, dim]` annotation syntax, label propagation
through elementwise/matmul/reduce/softmax, `reshape_labeled` and
`relabel` helpers, structured `EvalError::ShapeMismatch`,
label-aware `:vars`/`:describe`/trace JSON, and the "Named Axes"
tutorial lesson), every downstream saga (Tiny LM, LoRA, attention
variants, embedding viz) inherits shape-checked tensors. Saga 12
adds streaming dataset ops, a byte-level BPE tokenizer, and
`experiment "name"` objects with seed control and logged metrics.
