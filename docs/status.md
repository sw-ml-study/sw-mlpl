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

## Planned

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 12 | Tokenizers, datasets, experiment tracking | v0.9 | [ ] NEXT | 11.5 |
| 13 | Tiny language model end-to-end (CPU) | v0.10 | [ ] | 12 |
| 14 | MLX backend (Apple Silicon, lazy graph, fusion) | v0.11 | [ ] | 13 |
| 15 | LoRA / QLoRA / quantization | v0.12 | [ ] | 13 |
| 16 | Embedding visualization and RAG | v0.13 | [ ] | 13 |
| 17 | CUDA backend and distributed execution | v0.14 | [ ] | 14 |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | v0.15 | [ ] | 15 |
| 19 | LLM-as-tool REST integration | v0.16 | [ ] | 12 |

## Next saga to start

**Saga 12 -- Tokenizers, datasets, and experiment tracking.** Two
sagas shipped in rapid succession on top of Saga 11 (Model DSL):
11.5 added labeled shapes and structured shape errors (v0.7.5),
and the Compile-to-Rust saga added the `mlpl!` proc macro +
`mlpl build` native-binary CLI (v0.8.0, 9x measured speedup on a
100x100 reshape+reduce workload). Every downstream saga (Tiny
LM, LoRA, attention variants, embedding viz) now inherits both
shape-checked tensors and a path from MLPL source to a native
executable. Saga 12 adds streaming dataset ops, a byte-level BPE
tokenizer, and `experiment "name"` objects with seed control and
logged metrics -- the last surface-only saga before the Tiny LM
end-to-end.
