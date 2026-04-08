# MLPL Optimizers + Training Loop Saga (v0.6)

## Goal

Build proper training infrastructure on top of v0.5 autograd:
real optimizers (momentum-SGD, Adam) with state, learning-rate
schedules, a structured `train` loop construct, and richer
non-linear synthetic datasets (moons, circles). No model DSL,
no backend changes -- those are later sagas.

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

## What already exists

- `mlpl-autograd` reverse-mode tape with gradcheck-verified ops
- `param[shape]` / `tensor[shape]` constructors
- `grad(expr, wrt)` built-in
- v0.5 demos: tiny_mlp.mlpl, softmax_classifier.mlpl using grad
- `blobs`, `randn`, `one_hot`, manual SGD recipe `W <- W - lr * grad`

## Phases

### Phase 1: Optimizer state machinery
- New `mlpl-optim` crate (or module under `mlpl-runtime`).
- Optimizer state stored in the `Environment`, keyed by param name.
- `momentum_sgd(params, lr, beta)` and `adam(params, lr, b1, b2, eps)`
  built-ins that maintain per-param state across calls.
- Tests: state persists, momentum/Adam updates match reference math.

### Phase 2: Schedules
- `cosine_schedule(step, total, lr_min, lr_max)` built-in.
- `linear_warmup(step, warmup, lr)` built-in.
- Tests: schedule values at boundary points.

### Phase 3: Synthetic datasets
- `moons(seed, n, noise)` and `circles(seed, n, noise)` built-ins
  returning `[N, 3]` matrices `[x, y, label]` like `blobs`.
- Tests: shape, label balance, deterministic given seed.

### Phase 4: Training-loop sugar
- A `train { ... } for N steps` (or similar) statement that
  implicitly threads optimizer state and step counter.
- Captured loss vector available afterward via `last_losses`.
- Tests: end-to-end Adam-trained MLP on moons reaches >95% acc.

### Phase 5: Demos, tutorial, release
- New `demos/moons_mlp.mlpl`: 2-layer MLP trained with Adam on
  moons dataset, decision boundary rendered.
- New `demos/circles_mlp.mlpl`: same with circles.
- Tutorial lesson "Optimizers and schedules".
- Update `docs/saga.md`, `docs/status.md`, `docs/plan.md`,
  create `docs/milestone-optim.md`.
- Bump REPL banners to v0.6.
- Rebuild and deploy `pages/`.
- Tag `v0.6.0-optim`.

## Success criteria

- `momentum_sgd` and `adam` built-ins work on any `param[...]`.
- Adam-trained MLP on moons reaches >95% accuracy in <500 steps.
- `moons` and `circles` datasets ship with deterministic seeds.
- Training-loop sugar replaces the manual `repeat { grad; step }`
  pattern in at least one demo.
- Tutorial lesson on optimizers renders in the web REPL.
- All quality gates green, pages deployed, release tagged.
