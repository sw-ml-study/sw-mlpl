# Missing demos, missing docs, and the Saga 11 blocker

**Audit date:** 2026-04-09
**Scope:** tutorial coverage in `apps/mlpl-web/src/tutorial.rs`, README
coverage of language features, and an end-to-end smoke test of the
Saga 11 model DSL.

**Update 2026-04-09 (same day):** the "Blocker found during tutorial
drafting" section below has been *partially* resolved as part of
Saga 11 step `006-port-mlp-demos`. The grad tape now lowers
`apply(model_ident, X)` for `linear`, `chain`, and activation layers,
so `adam(loss_with_apply, mdl, ...)` trains end-to-end for MLP-shaped
models; `demos/moons_mlp.mlpl` and `demos/tiny_mlp.mlpl` have been
ported to the DSL and verified. Tape lowering for `residual`,
`rms_norm`, and `attention` is still TODO and remains a prerequisite
for step `007-transformer-block-demo`.

This document is not normative. It captures what is missing from
the user-facing surfaces after Sagas 8-11 landed so the next few
steps can be planned with eyes open.

## Demos present in `demos/` but not mentioned in `README.md`

- `demos/analysis_demo.mlpl`
- `demos/circles_mlp.mlpl`
- `demos/moons_mlp.mlpl`

The web REPL's demo picker also does not yet include `circles_mlp`
or the compact one-liner ports of `tiny_mlp` / `moons_mlp` that
Saga 11 phase 5 is supposed to produce.

## Language features used in demos but absent from `README.md`

### Autograd and optimizers (Sagas 9 and 10)

- `param[shape]` trainable-leaf constructor
- `grad(loss_expr, wrt)` reverse-mode autograd built-in
- `momentum_sgd(loss, params, lr, beta)`
- `adam(loss, params, lr, b1, b2, eps)` -- accepts a single param
  identifier, a list of identifiers, or a model identifier
- `cosine_schedule(step, total, lr_min, lr_max)`
- `linear_warmup(step, warmup, lr)`
- `train N { body }` loop construct with implicit `step` binding
  and an auto-captured `last_losses` vector

### Model DSL (Saga 11)

- `linear(in_dim, out_dim, seed)`
- `chain(layer_a, layer_b, ...)`
- `tanh_layer()`, `relu_layer()`, `softmax_layer()`
- `residual(inner)`
- `rms_norm(dim)`
- `attention(d_model, heads, seed)` -- multi-head self-attention
- `apply(model_identifier, X)` -- forward pass on a stored model

### Datasets and analysis helpers

- `moons(seed, n, noise)` synthetic dataset (Saga 10)
- `hist(values, bins)`
- `scatter_labeled(points, labels)`
- `loss_curve(losses)`
- `confusion_matrix(pred, truth)`
- `boundary_2d(surface, grid_shape, train_points, train_labels)`

### Web REPL surface polish (recent, undocumented)

- Tutorial mode with inline lessons and runnable examples
- Title badge (`docs/mlpl-badge.png`)
- Numeric output summarization with a collapsible `<details>`
  accordion and min/max/mean/median/std summary line

## Tutorial gaps in `apps/mlpl-web/src/tutorial.rs`

The current 19 lessons stop at "Optimizers and Schedules" and never
introduce the Saga 11 model DSL. The biggest gaps, in priority
order:

1. **Layered model DSL** -- `linear`, `chain`, activation layers,
   `rms_norm`, `residual`, `attention`, `apply`. All hand-rolled
   matmul recipes in the ML lessons could be rewritten once this
   lesson exists.
2. **Reductions and broadcasting** -- `reduce_add` axis semantics
   and scalar/vector/matrix broadcasting rules. Used in every ML
   lesson, never taught explicitly.
3. **Shape manipulation** -- `reshape`, `transpose`, `shape`,
   `rank`, `iota` as a dedicated lesson instead of as incidental
   setup inside the matrix/ML lessons.
4. **Synthetic datasets** -- `randn`, `blobs`, `moons` introduced
   as a group before they show up inside training demos.
5. **Decision boundaries as a first-class lesson** -- currently
   only name-dropped in "Visualizing Analyses."

Lesson 1 (the model DSL lesson) is step `008-modeldsl-tutorial-lesson`
in the Saga 11 plan, so it is already scheduled. Lessons 2-5 are
unscheduled nice-to-haves that would meaningfully improve the
onboarding path.

## Blocker found during tutorial drafting

While drafting the "Training a Layered Model" capstone lesson a
smoke test was run to confirm the end-to-end surface. It failed:

```mlpl
mdl = linear(2, 2, 1)
X = [[1.0, 0.0], [0.0, 1.0]]
Y = [[1.0, 0.0], [0.0, 1.0]]
grad(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)), mdl)
-- error: unsupported: grad: 'mdl' is not a tracked parameter
```

Two related problems:

1. **`apply` is not in the grad tape.** `crates/mlpl-eval/src/grad.rs`
   routes unary/binary ops through `Tensor` methods but has no
   arm for `apply`, so any loss expression that threads inputs
   through a model via `apply(mdl, X)` is opaque to reverse mode.
2. **`grad(loss, mdl)` on a model identifier is rejected.**
   `collect_params` recognizes a model identifier (and
   `env::model_params` returns the flat param name list), but
   `eval_grad` itself expects a single tracked param name. The
   adam path calls `eval_grad` per-leaf, so it inherits the same
   problem: even if you pass `mdl` to `adam`, the per-param
   gradient call is `grad(loss_with_apply, W_leaf)`, which still
   cannot see through `apply`.

The net effect is that the Saga 11 model DSL is **usable for
forward inference only**. Training with the DSL does not work
end-to-end today. That is why `demos/moons_mlp.mlpl` still inlines
the full forward pass inside its `adam(...)` call instead of
calling `apply(mdl, X)` -- the inlined form keeps every op on the
tape.

This directly blocks Saga 11 steps 006 and 007. Step 006's stated
success criterion is

> `adam(loss, params(model), ...)` works end-to-end

and step 007 ("tiny transformer block trains end-to-end inside
`train { }` and the loss decreases monotonically") cannot run
without a differentiable `apply`.

### What "implement soon" means here

Before continuing Saga 11, one of the following has to land:

- **Option A (preferred):** teach the grad tape to record `apply`.
  When the forward pass encounters `apply(mdl_ident, X)`, inline
  the model's structure onto the tape by walking `ModelSpec` and
  emitting the same primitive ops (`matmul`, `tanh`, softmax,
  layernorm, residual add, attention pieces) that
  `model_dispatch::apply_model` already runs eagerly. This keeps
  the user-facing surface unchanged and makes
  `grad(loss_with_apply, W_leaf)` work for every leaf the walker
  yields.
- **Option B (escape hatch):** add a `params(model)` built-in
  that returns an array-of-identifiers so users can pass
  `adam(..., params(mdl), ...)` and continue writing the forward
  pass manually inside the loss. This unblocks step 006 cosmetically
  but defeats the point of the DSL -- the demos would still carry
  the inlined loss novellas Saga 11 was chartered to delete.

Option A is the only path that honors the saga's success criteria.
It is one focused task: a new arm in the grad tape plus a
`ModelSpec`-to-tape lowering. It should land as an inserted step
before the current `006-port-mlp-demos`.

## Saga 11 remaining step count

`agentrail status` on 2026-04-09 shows 5 of 9 Saga 11 steps
complete:

| # | Step | Status |
|---|------|--------|
| 001 | model-value-and-linear | done |
| 002 | chain-and-activations | done |
| 003 | params-walker | done |
| 004 | residual-and-norm | done |
| 005 | attention-block | done |
| 006 | port-mlp-demos | **blocked on differentiable `apply`** |
| 007 | transformer-block-demo | blocked (same reason) |
| 008 | modeldsl-tutorial-lesson | blocked on 006/007 landing |
| 009 | modeldsl-release-v07 | waits on 008 |

So the headline number is **4 steps remaining**, but realistically
there is at least one new step (call it `005b-grad-apply` or
similar) that must be inserted before 006 can run. Treat Saga 11
as having **5 effective steps left**, not 4.

## Future sagas

`docs/plan.md` sections "Future saga sequence" and "Dependency
graph" list eight sagas beyond the current one. Sagas 9-13 must
run in order. Sagas 14-19 can reorder based on hardware and
interest.

| # | Saga | One-line goal |
|---|------|----------------|
| 12 | Tokenizers, datasets, experiment tracking | Streaming dataset ops, byte-level BPE, `experiment "name"` objects, reproducibility |
| 13 | Tiny LM end-to-end | Train a 1-5M param character/BPE LM on CPU in MLPL |
| 14 | MLX backend | Backend abstraction + lazy graph + kernel fusion on Apple Silicon |
| 15 | LoRA / QLoRA and quantization | `lora[rank=8] model`, 4-bit base, fine-tune the Saga 13 model |
| 16 | Embedding visualization and RAG | `embed`, pca/tsne/umap projections, RAG pipeline demo |
| 17 | CUDA backend and distributed execution | CUDA kernels for fused ops, `run model on nodes[...]`, homelab LAN training |
| 18 | Distillation, ICL/ICRL, engram memory | Distillation pipelines, trajectory updates, multi-model orchestration |
| 19 | LLM-as-tool integration | REST client built-ins, teacher-model distillation workflows, codegen helpers |

**Count:** eight future sagas are already planned. No additional
sagas beyond Saga 19 are currently documented; anything further is
research-frontier material in `docs/research.txt` /
`docs/research2.txt` and has not been shaped into a saga.
