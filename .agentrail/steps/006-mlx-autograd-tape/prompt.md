Phase 3 step 006: tape-lowered ops on MLX + gradcheck parity.

Make `grad(expr, wrt)` work inside a `device("mlx") { }`
block.

1. Decide during TDD between two paths:
   (a) Lean on `mlx-rs`'s `vjp` / `grad` helpers and wire
       them into the tape so the MLPL tape is essentially a
       thin wrapper.
   (b) Hand-write per-primitive backward passes on MLX that
       mirror the CPU tape.
   Write the decision and its rationale into a comment block
   at the top of the relevant module; document it in the step
   summary.
2. Ship MLX-backed backward paths for the existing tape
   primitives: `add`, `sub`, `mul`, `div`, `neg`, `exp`,
   `log`, `relu`, `tanh`, `sigmoid`, `softmax`, `sum`,
   `mean`, `matmul`, `transpose`, `reshape`.
3. Gradcheck each one against finite differences on Saga 9's
   fixtures. Parity test: `grad(expr, wrt)` on MLX matches
   the CPU path within the step 001 tolerance.
4. Integration test: `grad(cross_entropy(apply(model, X),
   Y), model_param)` on a Saga 13 Tiny LM slice produces the
   same gradient on CPU and MLX (within tolerance).
5. Quality gates + `/mw-cp`. Commit message references step 006
   and the decision made in (1).
