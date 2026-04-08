# MLPL Model DSL Saga (v0.7)

## Goal

Introduce composition primitives so neural-net models can be
expressed as data instead of long fused matmul expressions.
After this saga, the moons / circles / tiny_mlp demos should
read like model definitions, not loss-expression novellas, and
optimizers should walk a parameter tree instead of an explicit
identifier list.

No backend changes. No new training-loop sugar (Saga 10 already
gave us `train { }`). No tokenizer / dataset work (that is
Saga 12). The goal is purely surface-level: *how do you write
a model in MLPL?*

## Quality requirements (every step)

1. TDD: failing tests first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.

## What already exists (post v0.6)

- `mlpl-autograd` reverse-mode tape, `param[shape]` /
  `tensor[shape]` constructors, `grad(expr, wrt)` built-in
- `momentum_sgd`, `adam` optimizer built-ins with persistent
  per-param state on `Environment::optim_state`
- `cosine_schedule`, `linear_warmup` scalar schedules
- `train N { body }` construct with implicit `step` and
  `last_losses` capture
- `moons`, `circles`, `blobs` synthetic datasets
- Tutorial "Optimizers and Schedules" lesson

## Phases

### Phase 1: Model value type and atomic layers
- A new `Value::Model` (or similar) representing a callable
  layer / model with an attached parameter tree.
- Atomic layers: `linear(in, out, seed)` returning a model
  whose parameters are `W` (in x out) and `b` (1 x out).
- Apply syntax: pick a surface (`model(X)` function-call style
  vs. `apply(model, X)` built-in) and stick with it.

### Phase 2: Composition combinators
- `chain[layer1, layer2, ...]`: sequential composition.
- Activation layers `tanh_layer`, `relu_layer`,
  `softmax_layer` so chains can be pure data.
- `params(model)` walker returning a flat list of param
  identifiers usable as the second argument to
  `momentum_sgd` / `adam`.

### Phase 3: Skip connections and normalization
- `residual[block]`: y = x + block(x).
- `layer_norm(dim)` and/or `rms_norm(dim)` as a layer.
- Tests: residual identity at zero-init, norm preserves shape
  and zeroes the mean (layer norm) / unit RMS (rms norm).

### Phase 4: Attention block
- `attention[d_model, heads]` multi-head self-attention as a
  composable layer (Q/K/V projections + output projection).
- Reuse the existing softmax / matmul / transpose tensor ops.

### Phase 5: Demo refresh
- Port `demos/moons_mlp.mlpl` and `demos/tiny_mlp.mlpl` to
  the DSL: `chain[linear(2, 8), tanh_layer, linear(8, 2),
  softmax_layer]` style.
- New `demos/transformer_block.mlpl`: a tiny 2-layer
  transformer block (residual + norm + attention + MLP) on a
  toy sequence task.

### Phase 6: Tutorial, docs, release
- New "Model composition" tutorial lesson.
- Update `docs/saga.md`, `docs/status.md`, `docs/plan.md`,
  create `docs/milestone-modeldsl.md`.
- Bump REPL banners to v0.7.
- Rebuild and deploy `pages/`.
- Tag `v0.7.0-modeldsl`.

## Success criteria

- A 2 -> 8 -> 2 MLP can be written as a one-line `chain[...]`
  expression.
- `params(model)` returns a list that adam can consume directly:
  `adam(loss, params(model), ...)` works end-to-end.
- The ported moons_mlp demo is shorter than its v0.6 form
  (measured in non-blank source lines) without losing accuracy.
- A tiny transformer block trains end-to-end on a toy task
  inside `train { }` and the loss decreases monotonically over
  the run.
- Tutorial lesson on model composition renders in the web REPL.
- All quality gates green, pages deployed, release tagged.
