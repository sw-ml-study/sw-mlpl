Phase 1 step 003: reductions + softmax + log-softmax on MLX.

Close the forward-pass gap so the Tiny LM's forward can run on
MLX end-to-end.

1. Port to `mlpl-mlx`:
   - Reductions: `reduce_add`, `reduce_mul`, `mean`, `argmax`
     (axis-aware, label-aware; axis-name string arg supported
     where the CPU path supports it).
   - `softmax`, `log_softmax` (max-subtraction for numerical
     stability, same invariant as the CPU path).
   - `cross_entropy(logits, targets)` over integer `targets`.
2. Parity tests vs `mlpl-rt` on `[4, 3]` and `[2, 3, 5]`
   fixtures for reductions + softmax; `[B*T=8, V=5]` fixture
   with fixed integer targets for cross_entropy. Use the step
   001 tolerance.
3. Label propagation: reduced axis's label dropped; softmax
   preserves labels; cross_entropy returns an unlabeled scalar
   (matching CPU behaviour).
4. At the end of this step, a Saga 13 Tiny LM forward pass
   (embedding + positional + causal attention + rms_norm +
   linear + cross_entropy) runs through the MLX runtime and
   produces the same loss as the CPU path, within tolerance.
   Land a parity test that demonstrates this.
5. Non-Apple CI stays green.
6. Quality gates + `/mw-cp`. Commit message references step 003.
