# Milestone: Model DSL (v0.7)

Saga 11. Built composition primitives on top of the v0.6 optimizer
and training-loop infrastructure so neural-net models can be
expressed as data instead of long fused matmul expressions. After
this milestone, optimizers walk a parameter tree instead of a flat
identifier list, and the ported MLP / transformer demos read like
model definitions rather than loss-expression novellas.

## What shipped

### Model value type (mlpl-eval)
- New `Value::Model` variant carrying a callable layer description
  plus an attached parameter tree. Lives alongside `Value::Array`
  and the other runtime values; prints as a short tag in the REPL
  and is walkable by `:models` introspection.
- `apply(model, X)` built-in runs the forward pass. `apply` is
  also the entry point `grad()` uses when a tape walk sees a
  model on the left of a matmul / residual / norm / attention
  expression, so optimizers can differentiate straight through
  `apply(mdl, X)` without any identifier-list plumbing.

### Atomic layers
- `linear(in, out, seed)` -- `W : [in, out]`, `b : [1, out]`,
  deterministic init from the seed.
- Parameter-free activation layers: `tanh_layer()`, `relu_layer()`,
  `softmax_layer()`. These exist as first-class layers so a chain
  can be pure data instead of a mix of layers and raw ops.

### Composition combinators
- `chain(layer1, layer2, ...)` -- sequential composition. Accepts
  any number of child layers; nesting is fine.
- `residual(block)` -- skip connection: `y = x + block(x)`.
- `params(model)` -- flat walker returning every `param`
  identifier the model owns, suitable as the second argument to
  `momentum_sgd` / `adam`. In practice optimizers now accept the
  model directly and call `params()` internally, so
  `adam(loss, mdl, lr, b1, b2, eps)` is the common form.

### Normalization and attention
- `rms_norm(dim)` -- per-row RMS normalization, parameter-free.
  Tape-lowered to `x * rsqrt(mean(x*x))` where `rsqrt` is encoded
  as `exp(-0.5 * log(y))` so the existing autograd tape handles
  the backward pass without a new primitive.
- `attention(d_model, heads, seed)` -- multi-head self-attention
  with Q/K/V/output projections. `heads=1` is tape-lowered for
  end-to-end training; `heads>1` evaluates forward but returns
  `Unsupported` on the backward pass until per-head slicing
  lands in the tensor op surface.

### Ported demos
- `demos/moons_mlp.mlpl` and `demos/tiny_mlp.mlpl` rewritten as
  one-line `chain(linear, tanh_layer, linear, ...)` expressions
  trained inside `train 100 { adam(..., mdl, ...); loss }`.
  Source line counts dropped meaningfully relative to their v0.6
  form without losing accuracy on moons / blobs respectively.
- `demos/transformer_block.mlpl` -- new tiny 2-layer transformer
  block that stacks `residual(chain(rms_norm, attention))` and
  `residual(chain(rms_norm, linear, relu_layer, linear))` twice,
  followed by a final linear projection. Trains end-to-end on a
  random-input / random-target toy task with Adam for 100 steps;
  measured loss 143.87 -> 1.02 with a strictly monotonic decrease.

### REPL introspection
- `:vars`, `:models`, `:fns`, `:wsid`, `:describe` commands in
  both `mlpl-repl` and `mlpl-web`, so you can inspect the
  parameter tree of a model value from the REPL without parsing
  through `shape(params(mdl))` by hand.
- `:fns` split into user and built-in sections with `:help
  <topic>` dispatch for longer explanations.

### Tutorial
- New "Model Composition (the Model DSL)" lesson in
  `apps/mlpl-web/src/tutorial.rs` walking from a single
  `chain(linear, tanh_layer, linear)` MLP through `apply(mdl, X)`
  to Adam inside `train { }` on the moons dataset, with a
  decision-boundary render at the end.

## What did not change

- No backend changes -- still tree-walked CPU evaluation.
- No new training-loop sugar. Saga 10's `train N { body }` does
  the full job once you have a model value; the saga's goal was
  purely surface-level.
- No tokenizer / dataset work -- that is Saga 12.
- Multi-head attention is forward-only; the tape lowering for
  `heads > 1` needs per-head slicing primitives that are not
  yet in the tensor op surface.

## Notes for the next saga

Saga 11.5 (Named Axes and Shape Introspection) is the next
surface-only milestone: labeled shape metadata on `Value::Array`,
annotation syntax on assignment (`x : [batch, time, dim] = ...`),
label propagation through matmul/elementwise/reductions, and
structured `ShapeMismatch` errors. It is inserted between Saga 11
and Saga 12 because every later saga (Tiny LM, LoRA, attention
variants, embedding viz) benefits from labeled axes. See
`docs/milestone-named-axes.md` for the 10-step plan.

After 11.5 ships as v0.7.5, Saga 12 brings the streaming dataset
ops, byte-level BPE tokenizer, and `experiment "name"` tracking
that the tiny-LM saga depends on.
