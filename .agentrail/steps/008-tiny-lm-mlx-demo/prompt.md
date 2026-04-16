Phase 4 step 008: tiny LM MLX variant + bench harness.

Prove the speedup end-to-end.

1. New `demos/tiny_lm_mlx.mlpl` that is the Saga 13
   `demos/tiny_lm.mlpl` body wrapped in `device("mlx") {
   ... }`. No other source changes. Identical seeds, identical
   hyperparams, identical dataset.
2. Correctness: the loss curve shape matches the CPU demo
   within tolerance.
3. Extend `mlpl-bench` Criterion harness with an MLX row on:
   (a) the existing 100x100 reshape+reduce workload, and
   (b) a Tiny LM training-step workload (one forward +
       backward + Adam update on a realistic batch size).
4. Report both cold-start (first call, MLX compile overhead
   included) and warm timings. Go / no-go gate: >=5x warm-path
   wall-clock speedup over the interpreter CPU baseline on an
   M-class laptop. Target 10-50x per `docs/using-mlx.md`.
5. Write the measured numbers into `docs/benchmarks.md`.
6. If the `>=5x` gate fails, record the number, the likely
   bottleneck, and fall back: ship the MLX demo anyway with
   the honest number documented, and open a follow-up step
   for optimization. Do not block the saga on hitting 10x.
7. Wire the MLX demo into the web REPL demo list only if it
   runs there (MLX is not available under wasm32; if not,
   keep it CLI-only and note it in the tutorial).
8. Quality gates + `/mw-cp`. Commit message references step 008
   and quotes the measured speedup.
