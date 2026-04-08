# Autograd Milestone (v0.5.0)

This milestone makes differentiation a language primitive. A new
`mlpl-autograd` crate provides a reverse-mode tape on top of the
existing `DenseArray` substrate, the parser learns `param[shape]`
and `tensor[shape]` constructors, and a new `grad(expr, wrt)`
built-in lifts array expressions onto the tape and returns the
gradient with respect to a tracked parameter.

## Delivered

- [x] `mlpl-autograd` crate with reverse-mode `Tape` and `Tensor`
- [x] Tape ops: add, sub, mul, div, neg, exp, log, relu, tanh,
      sigmoid, softmax, sum, mean, transpose, reshape, matmul
- [x] Gradcheck tests against finite differences for every op
- [x] Parser surface for `param[shape]` and `tensor[shape]`
- [x] `grad(expr, wrt)` built-in via a tape-lifting mini-evaluator
- [x] `demos/tiny_mlp.mlpl` ported off hand-written backprop
- [x] `demos/softmax_classifier.mlpl` ported off hand-written backprop
- [x] Tutorial lesson "Automatic Differentiation" in the web REPL
- [x] REPL banners bumped to v0.5
- [x] `pages/` rebuilt and deployed
- [x] `v0.5.0-autograd` tag

## Acceptance criteria (all met)

- `mlpl-autograd` exists with gradcheck-verified ops
- `grad` works in the REPL on scalar and vector losses
- Tiny MLP and softmax classifier demos pass their existing
  accuracy integration tests with zero hand-written gradient math
- Tutorial lesson on autograd renders in the web REPL
- All quality gates green (cargo test, clippy -D warnings, fmt,
  sw-checklist)

## Demo: Linear regression with `grad`

```
X  = [[1],[2],[3],[4]]
Yc = [[2],[4],[6],[8]]
w  = param[1, 1]
lr = 0.05
repeat 100 { w = w - lr * grad(mean((matmul(X, w) - Yc) * (matmul(X, w) - Yc)), w) }
matmul(X, w)
```

Converges to slope 2 in 100 SGD steps with no hand-written
gradient formulas.

## What is intentionally not in v0.5

- Multi-parameter `grad` returning a tuple (single-wrt only for now)
- Optimizers beyond manual SGD (`W <- W - lr * grad`); Adam and
  schedules land in Saga 10
- A higher-level model DSL; combinators land in Saga 11
- Backend abstraction; everything still runs on the CPU
  `DenseArray` substrate

## Next

Saga 10 -- Optimizers and training loop. Builds Adam, momentum,
and learning-rate schedules on top of `grad`, plus a structured
training loop and richer synthetic datasets.
