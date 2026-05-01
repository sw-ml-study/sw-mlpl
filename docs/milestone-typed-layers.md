# Typed Layer Roles + Walked :describe Milestone (Saga 27)

## Why this exists

Sagas 23 and 26 give *values* a typed surface (Logit, Probability,
Loss, Weight, etc.). Sagas 24 and 25 give *distributions* and
*computation graphs* a typed surface. What remains untyped is the
single most user-facing structural object in MLPL today: the
`ModelSpec` tree returned by `linear`, `chain`, `residual`,
`attention`, `embed`, etc.

Today `:describe mdl` for a chain prints something like
"Model(Chain(2 hidden layers, 1 output layer))". Useful, but flat
-- it does not show per-layer input / output shapes, per-layer
input / output *types*, the parameter tree, or the activation
chain. A student inspecting a transformer block has to read the
code to understand the layers; the language tells them nothing.

Saga 27 promotes `ModelSpec` variants to typed roles ("Layer",
"ActivationLayer", "Optimizer", "Schedule", "Dataset") with
per-role metadata, and rewrites `:describe mdl` to walk the spec
tree printing each layer with its inferred typed input and output.
This is `typed-ml-concepts.md` Tier B condensed.

Goal ranking applied:

- **Educational** drives the saga. A student inspecting a
  transformer block should see a typed tree -- input shape,
  per-layer transformation, output shape -- without reading the
  source.
- **Maintainability** is served by unifying the layer kinds under
  a single typed-role mechanism, replacing the scattered
  per-variant `:describe` formatting that exists today.

## Non-goals

- New layer kinds. The set is the existing Saga 11 / Saga 13 /
  Saga 15 layers (Linear, Chain, Residual, RmsNorm, Attention,
  CausalAttention, Embed, Activation, LinearLora). New layer
  kinds land in their own sagas.
- Layer-tree editing. The role mechanism is read-only; the
  existing model constructors remain the only path to building a
  spec.
- Optimizer / Schedule / Dataset *runtime* changes. They become
  typed *roles* (carry metadata, render in `:describe`) but the
  ops themselves stay as they are.
- Static checking on `apply(mdl, X)`. A future static-typing
  saga can lift the input-shape predicate to lower time. Saga 27
  ships dynamic checks only.

## Quality requirements (every step)

Identical to Saga 23.

## What already exists

- `ModelSpec` enum in `mlpl-eval` with the variants listed above
  (Sagas 11, 13, 15).
- `Value::Model(ModelSpec)` runtime value (Saga 11).
- `params(mdl)` walker (Saga 11) -- the precedent for walking a
  spec tree.
- `apply(mdl, X)` differentiable forward (Saga 11). Saga 27 hooks
  into the first-call shape pinning that Saga 11.5 deferred.
- `:describe mdl` (Saga 11) -- the surface Saga 27 rewrites.
- Saga 23 `ValueTag` machinery for typing the input / output of
  each layer.

## Phases

### Phase 1: Layer role enum

A new `LayerRole` enum in `mlpl-eval`:

- `Layer { kind, input_kind, output_kind, params }` -- the most
  common case. `input_kind` and `output_kind` are Saga 23 tags.
- `ActivationLayer { kind }` -- parameter-free activations.
- `Embedding { vocab, d_model, table_param }` -- typed view of
  `embed`.
- `AttentionLayer { d_model, heads, masked }` -- typed view of
  `attention` / `causal_attention`.
- `Composite { kind, children }` -- typed view of `chain` /
  `residual`.

Every `ModelSpec` variant maps to exactly one `LayerRole`. The
mapping lives in a new `crates/mlpl-eval/src/layer_roles.rs`
module.

### Phase 2: First-apply shape pinning

Saga 11.5 deferred per-layer input / output label pinning on
`:describe mdl`. Saga 27 ships it.

- The first `apply(mdl, X)` records `X`'s labeled shape against
  `mdl`'s root and propagates through the spec tree, pinning
  each child's input / output shape.
- The pinned shapes live on a new `Environment::model_signatures:
  HashMap<String, ModelSignature>` side table.
- Subsequent `apply` calls with mismatched input shapes raise
  `EvalError::ShapeMismatch` (already a structured variant from
  Saga 11.5) but with a new tutoring hint pointing at the
  pinned signature.

### Phase 3: Walked :describe

Rewrite `:describe mdl` to use the role enum and the pinned
signature.

```
:describe transformer_block
transformer_block -- Composite(chain) -- 2 stages
  Layer 0: Composite(residual)
    Body: Composite(chain)
      RmsNorm[d=4]            in: Activation[seq, d=4]   out: Activation[seq, d=4]
      AttentionLayer[d=4, h=1] in: Activation[seq, d=4]  out: Activation[seq, d=4]
  Layer 1: Composite(residual)
    Body: Composite(chain)
      RmsNorm[d=4]            in: Activation[seq, d=4]   out: Activation[seq, d=4]
      Layer(linear)[in=4, out=16]  in: Activation[seq, d=4]   out: Activation[seq, d=16]
      ActivationLayer(relu)        in: Activation[seq, d=16]  out: Activation[seq, d=16]
      Layer(linear)[in=16, out=4]  in: Activation[seq, d=16]  out: Activation[seq, d=4]
  Params: 6 (W0, b0, W1, b1, W2, b2; total 192 weights)
  Frozen: 0
  Lora adapters: 0
```

The exact layout is iterated in step 003; the load-bearing
property is the per-layer typed input / output column.

### Phase 4: Hidden activation extraction

A new `:hidden mdl k` REPL command returns the Activation value
flowing out of layer `k` of the spec tree. Re-runs `apply` with
internal capture; returns the Saga 23-tagged Activation array.
Pairs with the Saga 25 ComputationGraph for "show me what
happened at layer 5".

### Phase 5: Typed Optimizer / Schedule / Dataset roles

Apply the same role-promotion treatment to three other
language-level concepts that today are only loosely typed:

- `Optimizer { kind, params, state }` -- a typed view of the
  Adam / momentum_sgd state map. `:describe` learns to render
  per-parameter optimizer state.
- `Schedule { kind, total_steps, current_step }` -- typed view
  of `cosine_schedule` / `linear_warmup`. `:describe` shows the
  curve and the current value.
- `Dataset { inputs, labels, n_classes, splits }` -- typed view
  of `moons` / `circles` / `blobs` / `load*` outputs.
  `:describe` shows the typed schema.

This phase ships the typed views (so `:describe` is uniformly
informative) and the metadata hooks (so future sagas can reason
about them) without changing the underlying ops.

### Phase 6: Tutorial lesson + retrospective + release

- New web REPL lesson "Typed Layer Tree" walks: build a
  transformer_block -> `:describe` shows the typed tree ->
  `apply` to pin shapes -> `:hidden` to extract a layer's
  output -> `:describe` of an Optimizer / Schedule / Dataset.
- Update `docs/using-typed-values.md` with a "Typed Layers"
  chapter.
- Update `docs/saga.md`, `docs/status.md`,
  `docs/are-we-driven-yet.md`.
- Bump REPL banners; rebuild `pages/`; tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | layer-role-enum            | 1 | `LayerRole` enum + `ModelSpec -> LayerRole` mapping |
| 002 | first-apply-shape-pinning  | 2 | `Environment::model_signatures` + propagation |
| 003 | walked-describe-mdl        | 3 | rewritten `:describe mdl` with typed columns |
| 004 | hidden-activation          | 4 | `:hidden mdl k` extraction command |
| 005 | typed-optimizer-schedule-dataset | 5 | three additional typed roles + `:describe` |
| 006 | typed-layers-tutorial      | 6 | new web REPL lesson |
| 007 | typed-layers-release       | 6 | docs, banners, pages rebuild, release tag |

Seven steps.

## Success criteria

- `:describe transformer_block` after one `apply` call shows the
  typed-tree layout from Phase 3 (or a close variant).
- `:hidden transformer_block 1` returns a `[seq, d_model]` array
  tagged `Activation[layer=transformer_block.linear_1,
  kind=relu]`.
- `:describe adam_state` after a training step shows per-param
  `m` / `v` moments with the right shapes and the current
  learning rate.
- `:describe moons_dataset` shows
  `Dataset[inputs=[N=200, dim=2], labels=[N=200], n_classes=2]`.
- Mismatched-shape second `apply` raises a tutoring error
  pointing at the pinned signature.
- All existing demos still pass; existing `:describe` users
  see strictly *more* information, not less.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **First-apply pinning robustness.** A model that legitimately
  takes different input shapes (rare, but possible -- e.g.
  variable sequence length) would fight the pinning. Mitigation:
  pinning records *symbolic* dim sizes when the input is
  labeled, so `[seq=*, d_model=4]` accepts any seq length.
- **Walked-describe size.** A 12-layer transformer prints 80+
  lines of `:describe`. Mitigation: a `:describe mdl --short`
  flag that collapses repeated chain bodies (e.g. "12x
  TransformerBlock"); on by default once the chain has more
  than 4 identical children.
- **Optimizer state granularity.** Adam state per param can be
  large (`m` and `v` arrays both of param shape). `:describe`
  for Optimizer shows shapes and a values preview, not the
  full arrays.
- **Hidden activation re-run cost.** `:hidden mdl k` re-runs
  `apply` from scratch, capturing layer `k`'s output. Acceptable
  for small models; for the 12-layer transformer it might be
  slow. Mitigation: a `:hidden mdl all` form that captures
  every layer in one pass and caches.
- **LinearLora rendering.** The Saga 15 LoRA adapter pair
  doubles the param count of every linear it wraps; the
  walked-describe must report adapter rank and base / adapter
  param counts separately so the LoRA structure is legible.
