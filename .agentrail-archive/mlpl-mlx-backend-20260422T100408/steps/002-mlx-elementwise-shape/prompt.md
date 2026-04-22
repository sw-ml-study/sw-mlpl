Phase 1 step 002: core elementwise + shape ops on MLX.

Fill out the forward-pass surface on the MLX runtime.

1. Port these primitives into `mlpl-mlx`, each matching the
   corresponding `mlpl-rt` signature:
   - Elementwise: `add`, `sub`, `mul`, `div`, `neg`
   - Activations (forward): `exp`, `log`, `tanh`, `sigmoid`,
     `relu`
   - Shape: `reshape`, `transpose`
2. Labels propagate identically to the CPU path -- `mlpl-mlx`
   does not own `LabeledShape`; it borrows `mlpl-core`'s and
   routes label propagation through the same helpers.
3. Tests per primitive:
   - Unit test on a small fixture (shape, values, and labels).
   - Parity test vs `mlpl-rt` on the same fixture, using the
     tolerance convention established in step 001.
4. Gradcheck / backward passes are NOT in scope here -- they
   land in step 006. Forward correctness only.
5. Non-Apple CI stays green (same feature gating as step 001).
6. Quality gates + `/mw-cp`. Commit message references step 002.
