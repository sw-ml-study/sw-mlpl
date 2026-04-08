# Milestone: Optimizers + training loop (v0.6)

Saga 10. Built proper training infrastructure on top of the
v0.5 autograd primitive: real optimizers with persistent state,
learning-rate schedules, a structured training loop construct,
and two non-linear synthetic datasets.

## What shipped

### Optimizers (mlpl-eval)
- `momentum_sgd(loss_expr, params, lr, beta)` -- heavy-ball
  momentum SGD with per-parameter velocity buffers stored on
  the evaluation environment.
- `adam(loss_expr, params, lr, b1, b2, eps)` -- standard Adam
  with bias correction and per-parameter first/second moment
  buffers, plus a per-optimizer step counter.
- Both surfaces accept either a single param identifier or an
  array literal of identifiers (`[W1, b1, W2, b2]`).
- State lives in a new `OptimizerState` map on `Environment`,
  keyed by `(optimizer_name, param_name, slot_name)`. Calling
  the same optimizer multiple times persists state across
  calls without any explicit threading from the program.

### Schedules (mlpl-runtime)
- `cosine_schedule(step, total, lr_min, lr_max)`: cosine decay
  between `lr_max` (at step 0) and `lr_min` (at step total),
  clamped outside the range.
- `linear_warmup(step, warmup, lr)`: linear ramp from 0 to
  `lr` over the first `warmup` steps, clamped at `lr` after.
- Both are pure scalar functions with no state.

### Datasets (mlpl-runtime)
- `moons(seed, n, noise)` -- two interleaving half-circles,
  the classic `make_moons` shape.
- `circles(seed, n, noise)` -- inner ring (radius 0.5, label 0)
  inside an outer ring (radius 1.0, label 1).
- Both return `[N, 3]` matrices in the same `[x, y, label]`
  layout as `blobs()`, deterministic given seed.

### Training-loop sugar (mlpl-parser, mlpl-eval)
- New `train N { body }` statement form, parallel to `repeat`.
- Inside the body the iteration index is bound to `step`.
- The value of the body's final expression on each iteration
  is captured into a 1-D array stored in the environment as
  `last_losses` after the loop. Non-scalar final values are
  mean-reduced for the loss curve.
- Replaces the manual `repeat { grad; manual update; record
  loss into mask }` recipe used in v0.5 demos.

### Demos
- `demos/moons_mlp.mlpl` -- 2 -> 8 -> 2 tanh MLP trained with
  Adam inside `train { }` on the moons dataset. Renders the
  decision boundary with `boundary_2d` over a 30x30 grid.
- `demos/circles_mlp.mlpl` -- companion demo on the circles
  dataset with a wider 2 -> 16 -> 2 hidden layer.

### Tutorial
- New "Optimizers and Schedules" lesson in
  `apps/mlpl-web/src/tutorial.rs` walking from a single-param
  Adam quadratic through `train { }` and `last_losses` to
  cosine and warmup schedule samples.

## What did not change

- No backend changes -- still tree-walked CPU evaluation.
- No model DSL or parameter trees -- params are still flat
  identifiers in the environment. That work belongs to Saga 11.
- `grad()`'s tape walker still doesn't support `reduce_add` /
  `one_hot` / cross-entropy directly, so the loss inside
  `adam()` calls in the demos uses MSE on `softmax(...) - Y`
  (same constraint as the v0.5 `tiny_mlp` demo).

## Notes for the next saga

Saga 11 (Model DSL) should give us a way to express
`chain[linear(2, 16), tanh, linear(16, 2), softmax]` as a
single value with named parameters, which would let demos drop
their long fused matmul expressions and let optimizers walk a
parameter tree instead of an explicit identifier list. The
`OptimizerState` keyed-tuple storage should generalize to
hierarchical parameter names without changes.
