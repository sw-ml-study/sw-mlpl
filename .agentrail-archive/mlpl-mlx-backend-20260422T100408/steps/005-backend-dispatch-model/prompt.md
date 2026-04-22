Phase 2 step 005: backend dispatch in the Model DSL.

Make `apply`/`params` route through the active backend.

1. `apply(model, X)` and `params(model)` dispatch through the
   MLX runtime when the surrounding scope is `device("mlx") {
   }` and the model's parameters live on MLX. A model owns its
   parameters on one device.
2. New builtins `to_device(x, "mlx")` / `to_device(x, "cpu")`
   copy a tensor (and, applied to a model, its parameters)
   between devices. Tests cover round-trip equality: `cpu ->
   mlx -> cpu` returns bit-for-bit the original tensor; `mlx ->
   cpu -> mlx` matches the original within tolerance.
3. Cross-device ops are an error with a clear message (e.g.
   `device mismatch: cpu matmul mlx`). This must surface as a
   typed `EvalError` variant, not a panic.
4. Parity tests: the Saga 11 models -- `linear`, `chain`,
   `residual`, `rms_norm`, `attention`, `causal_attention`,
   `embed`, `sinusoidal_encoding`, activations -- produce
   equivalent outputs (within tolerance) on both devices for
   fixed seeds and fixed input fixtures.
5. At the end of this step, a Saga 13 Tiny LM forward pass
   wrapped in `device("mlx") { apply(model, X) }` produces
   the same logits (within tolerance) as the CPU path.
6. Quality gates + `/mw-cp`. Commit message references step 005.
