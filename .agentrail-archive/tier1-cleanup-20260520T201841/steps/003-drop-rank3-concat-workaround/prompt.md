Tier 1 saga step 003: remove the per-batch workaround in the attention rank-3 path now that rank-3 concat-along-axis-N is available.

Saga 29's batched-attention tape lowering had to work around the missing rank-3 concat by stacking per-batch slabs along axis 0 (which axis 0 always supported) instead of using the natural concat-along-batch. The workaround lives in crates/mlpl-eval -- grep for 'batched' / 'rank-3' / 'per-batch' in grad.rs and model_dispatch.rs to find the relevant block.

Replace the workaround with a direct rank-3 concat using the axis the data actually wants. This is a refactor, not a behavior change: the output of the batched-attention forward is unchanged.

TDD:
- The existing single-head batched-attention test in crates/mlpl-eval/tests/ MUST still pass without modification.
- Add one new test that exercises the refactored path directly: a [B=2, T=4, d_model=8] input through attention(8, 1), output shape check + a spot-check on one element against the reference NumPy-style computation.

Quality gates: cargo test -p mlpl-eval; cargo clippy -p mlpl-eval --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower.