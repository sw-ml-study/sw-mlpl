Saga 29 step 007 (inserted prereq for the trained ViT demo):
tensor-take-indexing.

Why: docs/milestone-vit.md step 005 wants a trained ViT
demo on pets_tiny, which means each train step needs to
(a) pick out one image from the [200, 3, 64, 64] X tensor
to feed through the model, and / or (b) pull the CLS row
out of a [B, 17, 128] post-attention activation to feed the
classifier head. Today MLPL has no array-indexing primitive
beyond `scatter(buffer, idx, value)` (rank-1 write only) --
extracting an arbitrary axis-slice is not expressible.

This prereq ships `take(x, axis, idx) -> x_slice`, a
tape-differentiable indexing op that drops a single axis at
a single integer index. The follow-up step (008,
batch-aware-attention) extends `attention(...)` to rank-3
batched input so the trained demo (now step 009) can do a
proper [B, T, d] forward pass instead of an outer per-image
Python-style loop.

Scope (one PR / one step):

1. mlpl-array: `DenseArray::take(axis, idx) -> DenseArray`.
   Drops one axis. Both axis and idx are usize. Shape
   transforms as `[d0, ..., d_axis, ..., d_n] -> [d0, ...,
   d_{axis-1}, d_{axis+1}, ..., d_n]`. Per-axis labels
   propagate (the dropped axis's label is removed). Errors
   cleanly when axis out of range, idx out of range.

2. mlpl-autograd: new `NodeKind::Take { parent, orig_shape,
   axis, idx }`. Forward uses `DenseArray::take`. Backward
   scatters the upstream gradient back into a zero-filled
   array of the parent's shape, with the upstream gradient
   landing at `axis = idx`.

3. mlpl-autograd: `Tensor::take(axis, idx)` method on the
   tape -- one new method.

4. mlpl-eval / mlpl-runtime: `take(x, axis, idx)` builtin
   dispatch (rank-2 scalar args for axis + idx). Add to
   inspect_groups + docs/lang-reference.

5. mlpl-eval grad.rs: `take` arm in `eval_tensor_fncall`
   so `grad(sum(take(X, 0, 5) ...), W)` traces through the
   tape. Both `axis` and `idx` are eager-evaluated to
   scalars; the resulting tape node carries them as
   `usize`.

6. mlpl-eval device.rs: `NodeKind::Take` arm in the MLX
   re-run match. `take` is a pure indexing op so the CPU
   forward value can stand without a peer round-trip.

7. Tests (`crates/mlpl-eval/tests/take_tests.rs`):
   - take(X, 0, k) for X = [N, ...] -> shape [...] correct
     for several N, ranks.
   - take labels: dropped axis label is removed,
     surviving labels carry through.
   - take out-of-range axis -> clean error.
   - take out-of-range idx -> clean error.
   - take + sum + grad parity vs finite differences on a
     [4, 3] fixture taking row 2.
   - Gradient is zero in every position except the taken
     row (verifying scatter back-prop).
   - Round-trip: `take(load_preloaded("pets_tiny").X, 0, 5)`
     returns shape `[3, 64, 64]`.

8. Contracts (contracts/eval-contract README): add a
   subsection on `take` semantics + gradient.

9. docs/glossary.md: new `take (builtin)` entry between
   the existing `t-SNE` and `tape` entries (alphabetical).
   README count drift bumped.

Quality gates: cargo test (workspace), clippy -D warnings,
fmt, markdown-checker (touched docs), sw-checklist held or
lowered. /mw-cp checkpoint. Commit + push before agentrail
complete.

Out of scope (followups, not this step):
- Multi-index `take_multi(x, axis, [i, j, k, ...])` /
  `gather(x, indices)`. The trained demo only needs single
  -index; the multi-index variant can land if a demo
  surfaces a need for it.
- Range / slice `x[a..b]` shorthand. The trained demo
  can compose multiple `take` calls.
- Batched matmul `[B, M, K] @ [K, N]`. The trained demo
  can either loop over the batch with `take` or reshape
  `[B, M, K] -> [B*M, K]` and reshape back. Real rank-3
  matmul is a future saga.
- Step 008's batch-aware attention -- that lands in its
  own step, gated on this one.

Why this is one step, not folded into the demo: every new
tape op + cross-crate match site is a structural change
(same pattern as patchify / concat in step 005). The
trained demo (now step 009) is a long demo file plus its
adam/grad orchestration; mixing the indexing primitive
into the same commit doubles the review surface for no
gain.
