# Tier 1 alpha-leak cleanup saga

## Why this exists

Two findings from `docs/language-audit.md` are both small, high-
leverage, and already-tripping users in practice. They closed
out Saga 29 (ViT) as known limitations rather than as bugs to fix.
This saga retires both before MLPL accumulates more demos and
documentation on top of the broken behavior.

- **Audit finding #18 (`concat` axis restricted to `{0, 1}`).** The
  `mlpl-array::concat` implementation literally errors out if
  `axis > 1`. The user-visible message is a misleading
  `ShapeMismatch` rather than "unsupported axis." This was tripped
  during Saga 29 when joining rank-3 batched-attention outputs
  along the batch axis required a workaround in
  `mlpl-eval`'s attention-stack lowering. NumPy, PyTorch, and JAX
  have supported arbitrary axes since day one.

- **Audit finding #19 (multi-head attention has forward-only tape).**
  `attention(d_model, heads, seed)` for `heads > 1` runs through
  the forward pass but the autograd tape stops at the per-head
  split. Training a multi-head ViT looks like it works -- the
  loss drops a little and then plateaus -- because the gradient
  is implicitly zero on the per-head splits. The browser
  multi-head pets demo demonstrates this directly: 30 adam steps,
  loss drops a few hundredths, accuracy hovers around 0.5.

## Goals

- **Correctness.** Multi-head ViTs and any rank-3+ concat path
  produce correct gradients and shapes. Existing single-head
  tape behavior is unchanged.
- **No regression in demos.** All existing Saga 29 demos
  (single-head ViT, multi-head pets demos, attention overlay)
  still pass; the multi-head one now actually trains to a
  higher training accuracy than chance.
- **Educational visibility.** Where the fix changes
  user-observable behavior (the multi-head demo loss curve),
  the demo intro/takeaway are updated so a reader sees the
  difference.

## Non-goals

- **Replacing the audit's other critical-tier items.** The
  scripting cluster (#22, #24, #26, #28), `vmap` (#10),
  `gather` (#12), the closures-don't-differentiate refactor
  (#1), and the booleans-as-floats lift (#3) are separate
  sagas. This saga touches only #18 and #19.
- **Refactoring the autograd tape.** The multi-head tape fix
  lowers onto existing tape primitives (matmul, softmax,
  transpose, stack) -- the same primitives single-head
  already uses. No new tape node types are needed.
- **Adding new builtins.** Pure capability lift on existing
  surface.

## Dependencies

- The `Stack` tape op (Saga 29 step 008) is already in place.
  The multi-head tape lowering reuses it for the per-head join.
- The `concat` axis-N work touches `mlpl-array::concat` and the
  `copy_concat_rows` helper. The fix generalizes the existing
  rank-2 stride logic; no new infrastructure.

## What already exists

- `concat(a, b, axis)` for `axis in {0, 1}` on any rank, with
  full backward through the tape. (`mlpl-array/src/ops.rs:454`,
  `mlpl-eval/src/grad.rs:164`)
- `attention(d_model, heads, seed)` forward path for any `heads
  >= 1`. (`mlpl-runtime/src/builtins.rs`, model DSL surface)
- Single-head `attention` tape lowering through Q/K/V projection,
  scaled-dot-product, output projection. (`mlpl-eval/src/grad.rs`)
- Multi-head pets demos: `vit_multihead_quick.mlpl` (CLI),
  "Pets: multi-head ViT (quick + viz)" (web), "Pets: attention
  overlay (per-head)" (web).

## Steps

### Step 001 -- concat axis-N forward

Lift the `axis > 1` restriction in `mlpl-array::concat`. The
existing implementation already handles arbitrary rank for axis
0 and 1; generalize `copy_concat_rows` to copy along any axis
by computing the outer-stride / inner-stride product. TDD:
fixture inputs of rank 3 and rank 4 concatenated along axis 2
and axis 3 respectively, asserted shape + element-wise content.

### Step 002 -- concat axis-N backward

Generalize the autograd tape's concat backward node to split
the upstream gradient along the same axis used in the forward.
Today the backward in `mlpl-eval/src/grad.rs` assumes axis < 2.
TDD: gradcheck on a rank-3 concat along axis 2, finite-difference
parity within fp32 tolerance.

### Step 003 -- drop the rank-3 concat workaround in attention

Saga 29's batched-attention stack lowering uses a per-batch
workaround because rank-3 concat along the batch axis was not
available. Remove the workaround; replace with the now-direct
concat. Verify no regression on the single-head batched-attention
test fixture.

### Step 004 -- multi-head attention tape lowering

Lower `attention(d_model, heads > 1, seed)` onto the same tape
primitives as `heads = 1`: Q/K/V projection (linear), per-head
scaled-dot-product (matmul + transpose + scale + softmax +
matmul), stack along the head axis, output projection (linear).
Update `mlpl-eval/src/grad.rs` to dispatch multi-head through the
new lowering. The single-head fast path stays.

TDD: gradcheck on a small `attention(d=8, h=2)` fixture, finite-
difference parity. The headline integration test is the
`vit_multihead_quick.mlpl` demo's training accuracy: must reach
> 0.8 on the balanced 20-image subset after 100 adam steps
(today it stays ~0.5).

### Step 005 -- update multi-head pets demos

The browser "Pets: multi-head ViT" demo's takeaway is currently
calibrated against the plateaued-loss reality. After step 004
the demo actually trains; the takeaway needs to be rewritten to
reflect the new behavior. Same for the attention-overlay demo.
The CLI demo (`vit_multihead_quick.mlpl`) needs its expected-
accuracy comment refreshed.

### Step 006 -- close out

Update `docs/language-audit.md` findings #18 and #19 with a
`Fixed in:` line citing the commit SHAs from steps 001 and 004.
Refresh `docs/plan.md`'s Breaking-change candidates section to
move both findings to a "Shipped" subsection.
