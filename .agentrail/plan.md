# MLPL Autograd v1 Saga (v0.5)

## Goal

Make differentiation a language primitive. Land a reverse-mode
autograd engine on the existing `DenseArray` substrate, expose
`tensor[...]`, `param[...]`, and `grad` at the language level,
and port the v0.4 demos that currently hand-write their
gradients onto the new engine. No new backends, no optimizers
beyond SGD, no model DSL -- those are later sagas.

## Quality requirements (every step)

1. TDD: failing tests first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test` (all tests)
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.

## What already exists

- `mlpl-array` DenseArray, shape, reductions, matmul
- `mlpl-runtime` built-ins (softmax, argmax, blobs, random, ...)
- `mlpl-eval` tree-walking evaluator
- `mlpl-parser` with current surface syntax
- v0.4 demos: `tiny_mlp.mlpl`, `softmax_classifier.mlpl`, ...
  (both hand-rolling gradient formulas)

## Phases

### Phase 1: Autograd core (Rust)
- New crate `mlpl-autograd` with a reverse-mode tape.
- `Tensor` wrapper holding `{ value: DenseArray, grad: RefCell<Option<DenseArray>>, node: NodeId }`.
- Graph nodes for: add, sub, mul, div, neg, matmul, sum,
  mean, exp, log, relu, tanh, sigmoid, softmax, transpose,
  broadcast, reshape.
- `backward(root)` walks the tape in reverse topo order and
  accumulates gradients into leaf tensors marked `requires_grad`.
- Tests: gradcheck against finite differences for every op.

### Phase 2: Language surface
- Parser support for `param[shape]` and `tensor[shape]`
  constructors (trainable vs. non-trainable leaves).
- New built-in `grad(expr, wrt)` where `wrt` is a parameter or
  list of parameters; returns array(s) matching their shape.
- Evaluator wires array-valued operations through the autograd
  tape when any input is a tracked tensor.
- Tests: end-to-end `grad` calls on scalar and vector losses.

### Phase 3: Demo ports
- Rewrite `demos/tiny_mlp.mlpl` to use `param` + `grad` instead
  of hand-written backprop. Must still pass the existing
  "MLP beats linear on XOR" integration test.
- Rewrite `demos/softmax_classifier.mlpl` the same way. Must
  still reach > 95% accuracy on separable blobs.
- Keep SGD manual for now (`W <- W - lr * grad`).

### Phase 4: Tutorial, docs, release
- Add tutorial lesson "Automatic differentiation" walking from
  a scalar example to the MLP port.
- Update `docs/milestone-ml.md` (or new `milestone-autograd.md`),
  `docs/saga.md`, `docs/status.md`, `docs/plan.md`.
- Bump REPL banner to v0.5.
- Rebuild and deploy `pages/`.
- Tag `v0.5.0-autograd`.

## Success criteria

- `mlpl-autograd` crate exists with gradcheck-verified ops.
- `grad` built-in works in the REPL for scalar and vector losses.
- Tiny MLP and softmax classifier demos pass their existing
  accuracy tests with zero hand-written gradient math.
- Tutorial lesson on autograd renders in the web REPL.
- All quality gates green, pages deployed, release tagged.
