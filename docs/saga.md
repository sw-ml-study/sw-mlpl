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
