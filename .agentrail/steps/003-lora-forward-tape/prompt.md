Phase 1 step 003: forward + autograd for `LinearLora`.

Make `apply(lora_m, X)` compute the LoRA forward and make
`grad(..., lora_m)` produce gradients on A and B (and on W,
b if they are not frozen). Frozen names receive no optimizer
update (step 001's invariant); gradients still flow through.

1. Extend `apply_model` in
   `crates/mlpl-eval/src/model_dispatch.rs` to handle
   `ModelSpec::LinearLora`.
   Forward formula:
     `y = X @ W + (alpha / rank) * X @ A @ B + b`
   Implementation notes:
   - The `(alpha / rank)` scalar can be pre-computed once per
     apply; apply it via elementwise scalar multiply to
     the `X @ A @ B` result.
   - Base Linear still does its normal matmul + bias; the
     adapter contribution is additive.
   - Shape checks: A is `[in, r]`, B is `[r, out]`,
     resulting delta is `[X.rows, out]`, matches the
     `X @ W` shape.
2. Extend the autograd tape (`crates/mlpl-eval/src/model_tape.rs`)
   so LoRA forward is recorded as a sequence of primitive
   ops that grad already understands (matmul chain + scalar
   scale + add). Do NOT add a new tape primitive -- the
   forward is already expressible in the existing
   primitives, so the tape sees matmul + matmul + scalar
   mul + add + matmul + add and backward just works.
   Verify this empirically in the tests (gradcheck).
3. MLX path: the same `apply_model` extension handles both
   CPU and MLX when the existing `device` dispatch applies.
   Confirm that `device("mlx") { apply(lora_m, X) }`
   dispatches matmul + scalar-scale through mlpl-mlx and
   matches CPU within fp32 tolerance.
4. Frozen-param interaction: `adam(loss, lora_m, ...)` must
   update A and B on every step and must leave W and b
   unchanged iff they are in `env.frozen_params`. The
   `freeze(base)` before `lora(base, ...)` flow is the
   intended user pattern; step 004's demo will exercise it.
5. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/lora_forward_tape_tests.rs`:
   - Numerical forward: construct W=I, A=[[1,0],[0,1],
     [0,0],[0,0]], B=[[2,0,0,0,0,0,0,0],[0,3,0,0,0,0,0,0]],
     alpha=rank=2 so scale=1. Then
     `apply(lora_m, X=[[1,2,0,0]])` should equal
     `[[1,2,0,0]] + 1 * [[1,2,0,0]] @ A @ B`. Compute the
     expected values manually and assert bit-equality.
   - Zero-init B: before any training, forward matches
     base (already covered by step 002 but re-verify here
     to pin the tape path).
   - Gradcheck: small W, A, B, finite-difference against
     autograd-computed dL/dA and dL/dB within 1e-4.
     Crucially include a dL/dW check that is NON-ZERO
     (grad flows through W) but confirm that adam with
     frozen W leaves W unchanged across 3 steps.
   - Adapter training: after `freeze(base); student =
     lora(base, 2, 1.0, 0)` and 10 adam steps on a simple
     regression loss, A and B have moved, W and b are
     bit-identical to pre-training.
   - MLX parity (triple-gated): CPU vs MLX forward match
     within fp32 tolerance, optimizer-step values match
     within fp32 tolerance across 3 steps.
6. Quality gates + `/mw-cp`. Commit message references
   Saga 15 step 003.
