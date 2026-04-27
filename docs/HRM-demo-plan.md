# HRM demo plan

## Goal

Build a small Hierarchical Reasoning Model style demo for MLPL:
two recurrent modules running at different update rates, trained on
compact reasoning tasks such as maze path finding, Sudoku fragments,
and ARC-like grid transforms.

This should not try to reproduce published HRM numbers. The demo goal
is to show that MLPL is good at expressing a complete experiment as
auditable source: model definition, curriculum, training loop,
metrics, trace, and visualization in one small program.

## Why this is better than a notebook

- The experiment is a single replayable `.mlpl` file, not hidden cell
  state.
- Intermediate recurrent state can be traced and visualized with the
  same language constructs used for training.
- The REPL path supports fast probes: inspect high-level and low-level
  module activations after any training run.
- The compiled application path can package the trained solver and
  puzzle visualizer as a deterministic CLI artifact.
- Remote MLX keeps the heavy training loop on the Apple Silicon peer
  while the orchestrator/client stays portable.

## Repository strategy

Use the existing `softwarewrighter/viz-hrm-ft` repo as the home for
demo-specific code, data, run scripts, visualizations, fine-tuning
experiments, and artifacts. This plan is for adding an sw-MLPL track
to that repo, not for creating a new repo.

Suggested layout inside the existing repo:

- `src/` for Rust host app glue.
- `mlpl/` for `.mlpl` programs.
- `data/` for generated tiny puzzle sets.
- `artifacts/` ignored by default for checkpoints and traces.
- `README.md` with one REPL flow and one compiled-app flow.

Keep the core HRM-inspired language features in `sw-mlpl`; keep the
demo app, generated puzzles, comparison scripts, and trained artifacts
in the existing HRM repo.

## MLPL support needed

1. Recurrent model primitives.
   - `rnn_cell(in, hidden, seed)`
   - `gru_cell(...)` or a minimal gated cell
   - `scan(cell, X, h0)` for sequence unrolling
2. Named loop state.
   - A way to express `high` updated every `k` low steps.
   - A way to collect `high_trace` and `low_trace` per iteration.
3. Puzzle datasets.
   - `maze(seed, n, h, w)` with solution path labels.
   - `sudoku(seed, n, size)` for tiny boards first.
   - `grid_task(name, split)` for ARC-like transforms later.
4. Visualizations.
   - Maze/grid image SVG.
   - State heatmaps for high/low recurrent vectors.
   - Loss and exact-solve-rate curves.
5. Compile path.
   - Either lower the required recurrent/train surface to Rust, or
     create an embedded-interpreter compiled app first.

## Demo shape

### REPL/interpreter flow

```mlpl
task = maze(0, 128, 8, 8)
model = hrm_tiny(input_dim(task), 64, 16, 0)

device("mlx") {
  train 200 {
    pred = apply_hrm(model, task.X, 4)
    loss = cross_entropy(pred, task.Y)
    adam(loss, model, 0.001, 0.9, 0.999, 0.00000001)
    acc_metric = exact_path_accuracy(pred, task.Y)
    loss
  }
}

svg(task.X[0], "maze")
svg(hrm_trace(model), "heatmap")
loss_curve(last_losses)
```

### Compiled-app flow

```sh
mlpl build mlpl/hrm_maze.mlpl -o target/hrm-maze-demo
target/hrm-maze-demo --seed 0 --steps 200 --out artifacts/run.json
```

If full train lowering is not ready, the compiled app should embed the
interpreter and expose a stable CLI around the `.mlpl` program.

## Phases

1. CPU-only toy HRM on binary sequence tasks.
2. Maze dataset + SVG visualization.
3. MLX training with remote peer support.
4. Recurrent trace visualization.
5. Compiled app wrapper.
6. Update `softwarewrighter/viz-hrm-ft` with reproducible sw-MLPL run
   scripts.

## Acceptance tests

- A 4x4 or 8x8 maze task trains to better-than-baseline exact path
  accuracy within a short CPU run.
- The same source runs in `device("mlx") { ... }`.
- Trace output is deterministic for a fixed seed.
- The demo README has REPL and compiled-app commands.

## References

- HRM paper page: <https://huggingface.co/papers/2506.21734>
- arXiv record: <https://arxiv.org/abs/2506.21734>
- Existing sw project repo: <https://github.com/softwarewrighter/viz-hrm-ft>
