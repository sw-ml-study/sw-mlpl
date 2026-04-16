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

## Planned

| # | Saga | Target | Status | Depends on |
|---|------|--------|--------|------------|
| 13 | Tiny language model end-to-end (CPU) | v0.10 | [~] IN PROGRESS (step 008/009) | 12 |
| 14 | MLX backend (Apple Silicon, lazy graph, fusion) | v0.11 | [ ] | 13 |
| 15 | LoRA / QLoRA / quantization | v0.12 | [ ] | 13 |
| 16 | Embedding visualization and RAG | v0.13 | [ ] | 13 |
| 17 | CUDA backend and distributed execution | v0.14 | [ ] | 14 |
| 18 | Distillation, ICL/ICRL, engram memory, orchestration | v0.15 | [ ] | 15 |
| 19 | LLM-as-tool REST integration | v0.16 | [ ] | 12 |

## Next saga to start

**Saga 13 -- Tiny language model end-to-end (CPU).** Three
surface-only sagas shipped in rapid succession on top of Saga 11
(Model DSL): Saga 11.5 added labeled shapes and structured
errors (v0.7.5); the Compile-to-Rust saga added the `mlpl!` proc
macro + `mlpl build` native-binary CLI (v0.8.0, 9x measured
speedup); Saga 12 added filesystem IO (`load`, `--data-dir`
sandbox), dataset prep (`shuffle`/`batch`/`split`/`for`), a
byte-level BPE tokenizer (`tokenize_bytes`, `train_bpe`,
`apply_tokenizer`, `decode`), and experiment tracking
(`experiment "name" { ... }`, `:experiments`, `compare(a, b)`)
in v0.9.0 (plus a byproduct UTF-8 lexer fix that lets string
literals carry arbitrary Unicode). With both shape-checked
tensors and a full text-to-tokens pipeline in place, Saga 13
trains a tiny transformer-block LM end-to-end on a small
corpus: tokenize -> batch -> Model DSL with `attention` and
`residual` -> `adam` in a `train { }` loop -> sampling loop.
