# MLPL Multi-Saga Plan

This plan sequences the remaining work from `docs/research2.txt`
(the small-LLM / training-platform vision) against the current
state of the repo. Completed sagas are listed in `docs/saga.md`;
through v0.4 we have the language core, REPL, SVG viz, ML
built-ins, and six end-to-end demos, but every demo still carries
its own hand-written gradients. Everything downstream of autograd
depends on removing that limitation.

## Guiding principles

- Each saga delivers a visible user-facing win (a demo or a
  workflow), not just plumbing.
- Preserve the "thinking space" loop: train -> visualize ->
  adjust. Visualization is core syntax, not an afterthought.
- Backends (MLX, CUDA) come only after the CPU path is correct,
  tested, and expressive enough to be worth accelerating.
- Keep the cellular crate layout; new capabilities land as new
  crates, not as bulges in existing ones.
- TDD + quality gates per CLAUDE.md on every step.

## Completed sagas (context)

- **Saga -1** Repo scaffolding
- **Saga 0** Foundation and contracts
- **Saga 1** Dense tensor substrate v1 (Shape, DenseArray, reshape, transpose, indexing)
- **Saga 2** Parser and evaluator foundation
- **Saga 3** CLI and REPL v1
- **Saga 4** Structured trace v1 (JSON export)
- **Saga 5** Visual web viewer v1 -- DEFERRED
- **Saga 6** ML foundations (v0.2): sigmoid, tanh, pow, comparisons,
  axis reductions, mean, zeros/ones/fill, logistic regression demo
- **Saga 7** SVG visualization v1 (v0.3): `mlpl-viz`, `svg()`,
  scatter/line/bar/heatmap/decision_boundary, hist, scatter_labeled,
  loss_curve, confusion_matrix, boundary_2d, browser REPL inline SVG
- **Saga 8** ML demos (v0.4): random/randn/argmax/blobs/softmax/one_hot;
  k-means, PCA, softmax classifier, tiny MLP, attention pattern demos;
  tutorial lessons for each
- **Saga 9** Autograd v1 (v0.5): `mlpl-autograd` reverse-mode tape;
  `param[...]` / `tensor[...]` constructors; `grad(expr, wrt)` built-in;
  tiny MLP and softmax classifier demos ported off hand-written backprop;
  "Automatic differentiation" tutorial lesson
- **Saga 10** Optimizers + training loop (v0.6): `momentum_sgd` and
  `adam` built-ins with per-param state on the environment;
  `cosine_schedule` / `linear_warmup` scalar helpers; `moons` and
  `circles` synthetic datasets; new `train N { body }` construct
  with implicit `step` binding and `last_losses` capture;
  `moons_mlp` and `circles_mlp` demos; "Optimizers and Schedules"
  tutorial lesson

## Future saga sequence

### Saga 11 -- Model DSL (NEXT)
`chain[...]`, `residual[...]`, `norm[...]`, `attention[...]`
combinators. Parameter trees with named addressing. Rebuild the
tiny MLP and attention demos using the DSL; add a 2-layer
transformer block demo (still CPU, still tiny).

### Saga 11.5 -- Named axes and shape introspection
Labeled shape metadata on `Value::Array`, annotation syntax on
assignment (`x : [batch, time, dim] = ...`), label propagation
through matmul/elementwise/reductions, structured `ShapeMismatch`
errors, and agent-facing introspection commands (`:describe`,
`:vars`, `:models`, `:fns`). Inserted between Saga 11 and Saga 12
because every later saga (Tiny LM, LoRA, attention variants,
embedding viz) benefits from labeled axes. Surface-only milestone:
no new kernels, no new backends, CPU only. See
`docs/milestone-named-axes.md` for the 10-step plan.

### Saga 12 -- Tokenizers, datasets, and experiment tracking
Streaming/lazy dataset ops (`load`, `tokenize`, `shuffle`,
`batch`), a byte-level BPE tokenizer, and `experiment "name"`
objects with seed control, config snapshots, and logged metrics.
Reproducibility story lands here.

### Saga 13 -- Tiny LM end-to-end
Train a character-level or tiny-BPE language model (~1-5M params)
on a small corpus entirely in MLPL on CPU. Visualize loss,
sample generations, and attention maps. This is the first saga
that proves the platform thesis end-to-end.

### Saga 14 -- MLX backend
Backend abstraction layer (`backend <- mlx | cpu`), lazy
execution graph, kernel fusion for the hot ops. Re-run the tiny
LM on Apple Silicon and show a concrete speedup. CUDA deferred.

### Saga 15 -- LoRA / QLoRA and quantization
`lora[rank=8] model`, layer-scoped adapters, shared subspaces,
4-bit quantization for the frozen base. Fine-tune the Saga 13
model on a small instruction set.

### Saga 16 -- Embedding visualization and RAG
`embed`, `plot pca/tsne/umap`, 3D scatter via SVG projection,
nearest-neighbor links. RAG pipeline demo over a small corpus.
Builds directly on the existing viz stack.

### Saga 17 -- CUDA backend and distributed execution
CUDA kernels for the fused ops from Saga 14; `run model on
nodes[...]` primitives; device placement syntax. Homelab LAN
training demo.

### Saga 18 -- Distillation, ICL/ICRL, engram memory
Distillation pipelines, trajectory-based updates, engram-style
memory attachments, and multi-model orchestration
(`orchestrate[planner, executor, verifier]`). The research
frontier items from section 6 of `research2.txt`.

### Saga 19 -- LLM-as-tool integration
REST client built-ins, teacher-model distillation workflows,
codegen helpers. Intentionally last: secondary to the
"build your own model" story.

## Dependency graph (abbreviated)

```
9 autograd
  |-- 10 optimizers -- 11 model DSL -- 11.5 named axes -- 12 data+tracking -- 13 tiny LM
  |                                                                                |
  |                                                                                +-- 14 MLX
  |                                                                                +-- 15 LoRA
  |                                                                                +-- 16 viz/RAG
  |                                                                                +-- 17 CUDA/dist
  |                                                                                +-- 18 distill/ICRL
  |                                                                                +-- 19 LLM REST
```

Sagas 14-19 can reorder based on hardware access and interest;
9-13 (plus the inserted 11.5) must run in order.

## Start next: finish Saga 11, then Saga 11.5 -- Named axes

Saga 11 (Model DSL) is mostly shipped: model values, atomic and
composed layers, params walker, residual + rms_norm + attention,
differentiable `apply()` through the tape (Linear/Chain/Activation,
then also Residual/RmsNorm/single-head Attention as of step 007),
and ported MLP + transformer-block demos. Remaining Saga 11 work
is the tutorial lesson (done ahead of schedule on 2026-04-09) and
the v0.7.0 release.

After Saga 11 ships as v0.7.0, Saga 11.5 adds labeled shapes and
introspection -- the surface-only milestone surfaced by
`docs/are-we-driven-yet.md` as the single biggest agent-usability
gap. See `docs/milestone-named-axes.md` for the 10-step plan.
