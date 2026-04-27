# TRM demo plan

## Goal

Build a Tiny Recursive Model style demo: a very small network applies
the same update repeatedly to refine an answer on grid and puzzle
tasks. This is a better first reasoning demo than a full HRM because
the architecture is simpler, the model is smaller, and the recursive
loop maps cleanly onto MLPL source.

## Why this is better than a notebook

- The recursive update rule is visible as source instead of hidden in
  Python class state.
- Each recursion depth can be inspected with `:vars`, trace export, and
  heatmap visualizations.
- The same program can run as a REPL experiment, a remote MLX training
  job, and a compiled puzzle solver.
- Exact reproducibility matters for small-data reasoning; MLPL source
  keeps seeds, data generation, training, and metrics together.

## Repository strategy

Use the existing `softwarewrighter/train-trm` repo as the home for
demo-specific code, saved puzzle sets, comparison charts, and
artifacts. This plan is for adding an sw-MLPL track to that repo, not
for creating a new repo.

Suggested layout inside the existing repo:

- `mlpl/trm_grid.mlpl`
- `mlpl/trm_sudoku.mlpl`
- `src/main.rs` for compiled app wrapper
- `data/` for generated puzzle fixtures
- `artifacts/` ignored by default

## MLPL support needed

1. Recursive application primitive.
   - `recur(model, state, steps)` or a source-level pattern that does
     not require unrolled copy/paste.
2. Lightweight shared-weight model support.
   - Same layer parameters reused across recursion steps.
3. Grid task helpers.
   - `grid_onehot`, `grid_argmax`, `grid_accuracy`.
   - Tiny ARC-like generated tasks before external ARC data.
4. Step visualization.
   - `svg(pred_steps, "grid_sequence")`.
   - Per-depth loss table and convergence curve.
5. Compiler support.
   - Lower recursion or provide an embedded-interpreter compiled app.

## Demo shape

### REPL/interpreter flow

```mlpl
task = grid_task("copy_fill", 0, 256)
model = trm_tiny(input_dim(task), hidden=64, seed=0)

device("mlx") {
  train 300 {
    pred_steps = recur(model, task.X, 8)
    pred = last_row(pred_steps)
    loss = cross_entropy(pred, task.Y)
    adam(loss, model, 0.001, 0.9, 0.999, 0.00000001)
    acc_metric = grid_accuracy(pred, task.Y)
    loss
  }
}

svg(pred_steps, "grid_sequence")
loss_curve(last_losses)
```

### Compiled-app flow

```sh
mlpl build mlpl/trm_grid.mlpl -o target/trm-grid-demo
target/trm-grid-demo --task copy_fill --recursions 8
```

## Phases

1. Implement CPU toy recursion over vector states.
2. Add generated grid tasks and exact-match metrics.
3. Add MLX remote training.
4. Add recursion-step visualization.
5. Add compiled wrapper.
6. Update `softwarewrighter/train-trm` README so the sw-MLPL path is
   useful without this monorepo context.

## Acceptance tests

- Shared parameters are reused across recursion steps.
- A tiny generated grid task learns above a non-recursive baseline.
- Visualization shows at least input, intermediate, and final grids.
- Demo runs in REPL and through the compiled-app entry point.

## References

- TRM paper page: <https://huggingface.co/papers/2510.04871>
- arXiv record: <https://arxiv.org/abs/2510.04871>
- Existing sw project repo: <https://github.com/softwarewrighter/train-trm>
