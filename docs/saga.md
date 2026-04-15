# Saga

## Saga -1: Repo scaffolding (COMPLETE)
Repo compartmentalization scaffolding.

## Saga 0: Foundation (COMPLETE)
Foundation and contracts.

## Saga 1: Dense tensor substrate v1 (COMPLETE)
Shape, DenseArray, reshape, transpose, indexing.

## Saga 2: Parser and evaluator foundation (COMPLETE)
Lexer, AST, parser with precedence, AST-walking evaluator.

## Saga 3: CLI and REPL v1 (COMPLETE)
Working REPL with all v1 syntax, built-in functions, tracing.

## Saga 4: Structured trace v1 (COMPLETE)
TraceEvent/Trace types, evaluator instrumentation, JSON export.

## Saga 5: Visual web viewer v1 (DEFERRED)
Deferred to post-MVP. MVP uses CLI + JSON trace export.

## MVP
Sagas 0 through 4 complete. MVP ships with REPL + trace export.

## Saga 6: ML foundations (COMPLETE)
Sigmoid, tanh_fn, pow, comparison operators, axis reductions, mean,
array constructors (zeros, ones, fill), and the first end-to-end
logistic regression demo. Delivered v0.2.

## Saga 7: SVG visualization v1 (COMPLETE)
`mlpl-viz` crate, `svg(data, type[, aux])` built-in, diagram types
(scatter, line, bar, heatmap, decision_boundary), `grid()` helper,
high-level analysis helpers (hist, scatter_labeled, loss_curve,
confusion_matrix, boundary_2d), browser REPL inline SVG rendering,
and download button. Delivered v0.3.

## Saga 11: Model DSL (COMPLETE)
Composition primitives for neural-net models: a new `Value::Model`
runtime value, atomic `linear(in, out, seed)` layers,
parameter-free activation layers (`tanh_layer`, `relu_layer`,
`softmax_layer`), sequential `chain(...)` composition,
`residual(block)` skip connections, `rms_norm(dim)` normalization,
and `attention(d_model, heads, seed)` multi-head self-attention
(tape-lowered for `heads=1`, forward-only for `heads>1`). A
`params(model)` walker returns the flat parameter list, and
optimizers now accept a model identifier directly so
`adam(loss, mdl, lr, b1, b2, eps)` trains every weight the model
owns through differentiable `apply(mdl, X)`. The v0.6 `moons_mlp`
and `tiny_mlp` demos were rewritten as one-line `chain(...)`
expressions, and a new `transformer_block.mlpl` demo stacks
`residual(chain(rms_norm, attention))` and `residual(chain(
rms_norm, linear, relu_layer, linear))` twice to train a tiny
2-layer transformer block end-to-end (loss 143.87 -> 1.02 over
100 Adam steps, strictly monotonic). New REPL introspection
commands -- `:vars`, `:models`, `:fns`, `:wsid`, `:describe` --
ship in both `mlpl-repl` and `mlpl-web`, and a new "Model
Composition" tutorial lesson walks from a single `linear` through
a chain MLP to Adam inside `train { }` on the moons dataset.
Delivered v0.7. See `docs/milestone-modeldsl.md`.

## Saga 10: Optimizers + training loop (COMPLETE)
Built proper training infrastructure on top of Saga 9 autograd.
New built-ins: `momentum_sgd(loss, params, lr, beta)` and
`adam(loss, params, lr, b1, b2, eps)` with per-parameter state
held in an `OptimizerState` map on the evaluation environment;
`cosine_schedule(step, total, lr_min, lr_max)` and
`linear_warmup(step, warmup, lr)` pure scalar schedules; and
two non-linear synthetic datasets `moons(seed, n, noise)` and
`circles(seed, n, noise)` returning `[N, 3]` `[x, y, label]`
matrices in the same layout as `blobs`. New `train N { body }`
language construct binds the iteration index to `step` inside
the body and captures each iteration's final value into a
`last_losses` 1-D array, replacing the manual
`repeat { grad; manual update; record loss }` recipe. Two new
demos -- `demos/moons_mlp.mlpl` and `demos/circles_mlp.mlpl` --
train a tanh MLP with `adam` inside `train { }` and render the
decision boundary with `boundary_2d`. New "Optimizers and
Schedules" tutorial lesson added to the web REPL. Delivered v0.6.

## Saga 9: Autograd v1 (COMPLETE)
Reverse-mode autograd as a language primitive. New `mlpl-autograd`
crate provides a tape-based `Tensor` with backward over add, sub,
mul, div, neg, exp, log, relu, tanh, sigmoid, softmax, sum, mean,
transpose, reshape, and matmul, all gradcheck-verified against
finite differences. Parser surface adds `param[shape]` and
`tensor[shape]` constructors, and a new `grad(expr, wrt)` built-in
lifts array expressions onto the tape and returns the gradient
with respect to a tracked parameter. The v0.4 `tiny_mlp` and
`softmax_classifier` demos were ported to use `param + grad`
instead of hand-written backprop, and a new "Automatic
Differentiation" tutorial lesson walks from a scalar minimization
to a one-layer linear regression. Delivered v0.5.

## Saga 8: ML demos (COMPLETE)
Synthetic data primitives (random, randn, argmax, blobs) and
higher-level ML built-ins (softmax, one_hot), plus six demos wired
into the browser REPL: k-means clustering, PCA via power iteration,
a linear softmax classifier, a tiny MLP on XOR-style data, and a
scaled dot-product attention pattern. Tutorial lessons added for
each. Delivered v0.4.

## Saga 11.5: Named Axes and shape introspection (COMPLETE)
Axis-labeled shapes threaded through `Value::Array` via a new
`LabeledShape` type on `mlpl-core`. `label(x, [...])` and `relabel(x,
[...])` primitives, annotation syntax on assignment (`x : [batch,
time, dim] = ...`), and `reshape_labeled(x, dims, labels)` as the
opt-in re-labeling path. Label propagation through elementwise
ops (with one-None-one-Some accepted and mismatches rejected),
matmul (contraction axis validated, outer dims passed through),
reduce/argmax (reduced axis's label dropped), and `map()`
(preserves labels through every math builtin). `reduce_add`,
`reduce_mul`, `argmax`, and `softmax` accept an axis name string
in place of an integer. Structured `EvalError::ShapeMismatch
{ op, expected: LabeledShape, actual: LabeledShape }` at the
evaluator boundary, with Display rendering as `op: expected
[seq=N, d=M], got [time=N, d=M]`. Label-aware `:vars` and
`:describe` (using `LabeledShape` Display), and trace JSON that
round-trips axis labels for labeled arrays and omits the key
entirely for unlabeled ones. New "Named Axes" tutorial lesson in
the web REPL; "Model Composition" lesson now annotates X as
`[batch, feat]` so labels flow through `apply(mdl, X)`. Delivered
v0.7.5. See `docs/milestone-named-axes.md`.

## Future: Compile-to-Rust (PLANNED, not scheduled)
Exploratory direction for lowering MLPL to Rust source. Three
targets from one codegen backend: an `mlpl!` proc macro for
embedding MLPL in Rust apps, a `mlpl build foo.mlpl` subcommand
that emits native Mac/Linux binaries via cargo+rustc, and a
leaner WASM path that reuses the same pipeline. Depends on
Saga 11.5 (LabeledShape is the compile-time key for static
einsum-class dispatch) and on MLPL keeping no `exec(string)` /
no runtime-code primitive. See `docs/milestone-compile-to-rust.md`.
