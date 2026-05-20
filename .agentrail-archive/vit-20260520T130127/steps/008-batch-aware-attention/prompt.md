Saga 29 step 008 (inserted prereq for the trained ViT demo):
batch-aware-attention.

Why: `attention(d, h, seed)` today requires rank-2 `[seq,
d_model]` input -- both `apply_attention` (Saga 11) and
`attention_weights` (Saga 13) reject rank-3 with a clean
error. The trained ViT demo (now step 009) wants to forward
a `[200, 17, 128]` batch through one attention head and back
out `[200, 17, 128]` activations. Without batch-aware
attention the demo's only option is a per-image Python-style
loop, which makes 100 train steps mean 20,000 forward passes
and is wasteful given the underlying matmul is already
broadcast-friendly.

This prereq extends the attention forward and tape lowering
to accept rank-3 batched input, applying single-head
attention independently per batch entry. Multi-head (`h > 1`)
is still rejected -- that lands in step 010 (multi-head tape
lowering) per docs/milestone-vit.md.

Scope (one PR / one step):

1. mlpl-eval `model_dispatch.rs` `apply_attention`: when the
   input is rank-3 with `dims[-1] == d_model`, accept the
   call. Internal: split into `dims[0]` batch entries, run
   the existing rank-2 forward on each, stack back. Avoid
   `take` on the input -- chunk the underlying data directly
   for efficiency. Output shape: `[B, T, d_model]`.

2. mlpl-eval `model_dispatch.rs` `attention_weights`: same
   rank-3 acceptance. Output for single-head is `[B, T, T]`.

3. mlpl-eval `model_tape.rs`: per-batch tape lowering. For a
   rank-3 input, emit B chains of the rank-2 lowering
   sequence (project Q/K/V, scaled dot-product, softmax,
   recombine), then stack the per-batch outputs via concat
   along axis 0. The output tape node has shape `[B, T, d]`.

4. mlpl-eval `grad.rs`: integrate the rank-3 path so
   `grad(loss, W)` walks through the batched forward.
   Hopefully no change here -- the model_tape change is
   what does the work.

5. Multi-head `h > 1` still rejects with the existing error.
   Step 010 will replace that arm.

6. Tests (`crates/mlpl-eval/tests/batch_attention_tests.rs`):
   - rank-2 `[seq, d_model]` input still works (regression).
   - rank-3 `[B, T, d_model]` input returns `[B, T, d_model]`.
   - For a `[2, 3, 4]` input, the per-batch output agrees
     entry-by-entry with calling rank-2 attention on each
     row independently (within fp tolerance).
   - `attention_weights(model, X)` accepts rank-3 input and
     returns `[B, T, T]`.
   - Gradcheck parity: `grad(sum(apply(mdl, [B, T, d])), Wq)`
     matches finite differences on a tiny `[2, 3, 4]`
     fixture.
   - Existing `attention_tests.rs` continues to pass.

7. Contracts (`contracts/eval-contract` README): update the
   attention layer subsection (if present) to document the
   rank-2 and rank-3 paths, and the still-rejected
   multi-head + rank-3 case.

8. docs/glossary.md / docs/lang-reference.md: update the
   `attention()` entry to mention the rank-3 batched form.

Quality gates: cargo test (workspace), clippy -D warnings,
fmt, markdown-checker, sw-checklist held or lowered.
/mw-cp checkpoint. Commit + push before agentrail complete.

Out of scope (followups):
- Multi-head h > 1 (lands in step 010 multi-head tape
  lowering).
- Higher-rank inputs (rank-4 `[B1, B2, T, d]`). Add only if
  a demo surfaces a need.
- Batched matmul as a general primitive. The attention
  path can internally loop; a public `matmul([B, M, K], [K,
  N]) -> [B, M, N]` is a separate followup if performance
  matters.

Why this is one step, not folded into the demo: same
pattern as steps 002 / 007 -- the prereq is a structural
change to the attention layer touching multiple cross-crate
sites; the demo wants a single readable training loop.
