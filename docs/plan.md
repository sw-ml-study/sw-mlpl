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
- **Saga 11.5** Named axes and shape introspection (v0.7.5):
  `LabeledShape` on `mlpl-core`; `label` / `relabel` /
  `reshape_labeled` builtins; `x : [batch, dim] = ...` annotation
  syntax; label propagation through elementwise, matmul,
  reduce/argmax, and `map()`; axis-name arg for
  `reduce_add`/`reduce_mul`/`argmax`/`softmax`; structured
  `EvalError::ShapeMismatch` with op-aware Display; label-aware
  `:vars`/`:describe` and trace JSON; "Named Axes" tutorial
  lesson and labeled Model Composition example
- **Compile-to-Rust saga** (v0.8.0): `mlpl-rt` runtime target,
  `mlpl-lower-rs` AST -> `TokenStream` codegen (scalar, array,
  fncall, variable, labels, matmul with static contraction
  check), `mlpl-macro` proc-macro, `mlpl` facade crate with
  hidden `__rt` re-export, `mlpl-build` CLI that compiles a
  `.mlpl` file to a native binary (and cross-compiles to
  wasm32-unknown-unknown), `mlpl-parity-tests` harness proving
  bit-for-bit agreement on 9 programs. Measured 9.05x speedup
  on a 100x100 reshape+reduce workload. See
  `docs/milestone-compile-to-rust.md`.
- **Saga 11** Model DSL (v0.7): `Value::Model` runtime value;
  `linear(in, out, seed)` atomic layer; parameter-free
  `tanh_layer` / `relu_layer` / `softmax_layer` activations;
  `chain(...)` sequential composition; `residual(block)` skip
  connections; `rms_norm(dim)` normalization; `attention(
  d_model, heads, seed)` multi-head self-attention (tape-lowered
  for `heads=1`); `params(model)` walker and optimizer
  integration so `adam(loss, mdl, ...)` walks the parameter
  tree; differentiable `apply(mdl, X)`; `moons_mlp` and
  `tiny_mlp` demos ported to one-line `chain(...)` form; new
  `transformer_block.mlpl` demo (loss 143.87 -> 1.02 over 100
  Adam steps); `:vars` / `:models` / `:fns` / `:wsid` /
  `:describe` REPL introspection commands; "Model Composition"
  tutorial lesson. See `docs/milestone-modeldsl.md`.

## Future saga sequence

### Saga 11.5 -- Named axes and shape introspection (COMPLETE, v0.7.5)
Shipped: `LabeledShape` type on `mlpl-core`; `label(x, [...])` /
`relabel(x, [...])` / `reshape_labeled(x, dims, labels)` builtins;
`x : [batch, time, dim] = ...` annotation syntax; label propagation
through elementwise (with `merge_labels`), matmul (contraction
axis validated, outer dims threaded), reduce/argmax (reduced
axis's label dropped), and `map()` (preserves through math
builtins); axis-name arg for `reduce_add` / `reduce_mul` /
`argmax` / `softmax`; structured `EvalError::ShapeMismatch { op,
expected, actual }` with op-aware Display; labels in `:vars` /
`:describe` (via `LabeledShape` Display) and in the trace JSON
export (serde-skip-when-None); "Named Axes" tutorial lesson and
labeled `apply(mdl, X)` in the Model Composition lesson.
Deferred: per-layer input/output label pinning on `:describe
<model>` (needs `Environment` signature tracking on first
`apply` -- a natural follow-up).

### Compile-to-Rust saga (COMPLETE, v0.8.0)
Shipped: `mlpl-rt` runtime target, `mlpl-lower-rs` (path-
configurable AST -> Rust TokenStream codegen), `mlpl-macro`
proc-macro (`mlpl! { ... }` emits `compile_error!` on lower
failure), `mlpl` facade crate, `mlpl-build` CLI
(`mlpl-build foo.mlpl -o foo [--target <triple>]`),
`mlpl-parity-tests` harness. Static matmul contraction check
at lower time; runtime label checks still catch anything the
static pass cannot resolve. 9.05x measured speedup on a 100x100
reshape+reduce workload. Deferred: `TensorCtor`, `Repeat`,
`Train`, autograd, optimizers, Model DSL -- they need tape-
state or loop lowering and sit in a follow-up.

### Saga 12 -- Tokenizers, datasets, and experiment tracking (NEXT)
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

## Start next: Saga 12 -- Tokenizers, datasets, experiment tracking

Saga 11.5 (Named axes and shape introspection) shipped as v0.7.5
(tag `v0.7.5-named-axes`). Compile-to-Rust shipped as v0.8.0
(tag `v0.8.0-compile-rs`): `mlpl!` proc macro, `mlpl-build` CLI
with cross-compile, parity harness with 9x measured speedup,
four new crates (`mlpl-rt`, `mlpl-lower-rs`, `mlpl-macro`,
`mlpl`) plus a dev-only parity-tests crate. See
`docs/milestone-compile-to-rust.md` for the retrospective.

Saga 12 closes the last surface-only gap before the Tiny LM:
streaming/lazy dataset ops, a byte-level BPE tokenizer, and
`experiment "name"` objects for reproducible runs. Labeled
shapes from Saga 11.5 carry through into the dataset pipeline
so `batch : [B, T]` annotations stay meaningful all the way to
the training loop.
