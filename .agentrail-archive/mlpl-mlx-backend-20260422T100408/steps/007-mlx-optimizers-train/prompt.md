Phase 3 step 007: optimizers + train { } on MLX.

Close the training loop on MLX.

1. `adam(loss, model, ...)`, `momentum_sgd(loss, model, ...)`,
   `train N { body }`, `last_losses`, `cosine_schedule`,
   `linear_warmup`, and `experiment "name" { body }` all work
   unchanged inside `device("mlx") { }`. The
   `OptimizerState` map entries allocated inside an MLX block
   hold MLX-typed tensors.
2. `_metric` capture and `ExperimentRecord` logging work
   identically -- the on-disk `run.json` has no knowledge of
   which device ran the loop.
3. Parity test: one Adam step (same seed, same hyperparams,
   same batch) on the Saga 11 `tiny_mlp` fixture produces
   parameter updates that match CPU within tolerance.
4. Parity test: `train 3 { adam(...) }` on a Saga 13 Tiny LM
   micro-slice produces the same `last_losses` vector on CPU
   and MLX within tolerance.
5. Non-Apple CI stays green -- MLX-feature tests remain
   feature- and target-gated.
6. Quality gates + `/mw-cp`. Commit message references step 007.
