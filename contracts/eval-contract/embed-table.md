# `embed_table` Contract (Saga 16.5 step 002)

## Purpose

`embed_table(model) -> [vocab, d_model]` returns the
lookup-table parameter of the first Embedding layer
found in `model`'s spec tree. Closes the Saga 16 gap
where training a full `chain(embed, transformer_block,
head)` had no source-level way to pull the learned
embedding weights back out; `apply(standalone_embed,
iota(V))` only worked when the embed was a top-level
standalone model.

## Signature

```
embed_table(model) -> table
```

- `model` -- bare model identifier (looked up in
  `env.models`) or any expression that evaluates to
  `Value::Model`. Same argument shape as `clone_model`,
  `freeze`, `unfreeze`, and `lora`.
- `table` -- rank-2 float array `[vocab, d_model]`
  cloned from `env`. Subsequent mutations of `env`'s
  parameter do not retroactively mutate the returned
  array, and vice versa; the returned value is a
  snapshot.

## Tree walk semantics

Depth-first left-to-right walk of the `ModelSpec`:

- `Embedding { table, .. }`: return the `[vocab,
  d_model]` matrix bound to `table` in `env`.
- `Chain(children)`: iterate `children` in order;
  return the first child whose recursive walk succeeds.
- `Residual(inner)`: recurse into `inner`. (`Residual`
  wrapped around an Embedding is semantically weird at
  apply time -- embed changes input shape, so the
  `x + embed(x)` summation is ill-typed -- but the spec
  tree is well-formed and `embed_table` does not apply
  the model, so the walk still works.)
- `Linear`, `Activation`, `RmsNorm`, `Attention`,
  `LinearLora`: no Embedding; skip.

If no Embedding layer is found anywhere in the spec
tree, `embed_table` returns
`EvalError::Unsupported("embed_table: model contains
no Embedding layer")`.

## First-match semantics

If a model somehow has two Embedding layers in its
spec tree (not a shipped pattern; encoder/decoder
stacks with separate embeddings are out of Saga 16.5's
scope), `embed_table` returns the first one encountered
in depth-first left-to-right order.

Callers who need a specific Embedding by path should
construct the sub-model directly and call
`embed_table` on it. A future saga can add a
path-selector variant (`embed_table(model,
"encoder.embed")`) once a multi-embedding use case
surfaces.

## Determinism

Bit-identical outputs for two calls on the same model
in the same `env`. The walk is purely structural and
uses no PRNG.

## Error cases

All errors surface as
`EvalError::Unsupported(String)` or
`EvalError::BadArity { func: "embed_table", ... }`.

- **Wrong arity.** Anything other than 1 argument.
- **Non-model argument.** "embed_table: 'x' is not a
  model" (for bare identifiers) or "embed_table:
  argument must evaluate to a model" (for expressions
  that evaluate to a non-`Value::Model`).
- **Model has no Embedding layer.** "embed_table:
  model contains no Embedding layer".

## What this contract does NOT cover

- **Multi-embedding support.** First-match only.
  Selectors like `embed_table(model, "encoder.embed")`
  are a follow-up.
- **Path or index-based introspection of other
  sublayers.** Not a general-purpose model-introspection
  API; returns only the Embedding lookup table.
- **In-place mutation of the embed table.** The return
  value is a clone. Writing to it does not affect the
  original model's parameter; to update the weights,
  train the model via `adam` / `momentum_sgd` and call
  `embed_table` again to re-snapshot.
- **Returning the embedding parameter name.** The
  returned value is the table itself; the string name
  (used internally by `ModelSpec::Embedding { table,
  ... }`) is not exposed.
