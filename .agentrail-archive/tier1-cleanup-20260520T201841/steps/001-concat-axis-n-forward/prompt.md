Tier 1 saga step 001: lift the axis > 1 restriction in mlpl-array::concat forward.

Today crates/mlpl-array/src/ops.rs:469 returns ShapeMismatch if axis > 1, even though the rest of the function already handles arbitrary rank for axis 0 and 1. Generalize copy_concat_rows so it can copy along any axis by computing outer-stride * inner-stride from the input shape.

TDD (Red/Green/Refactor):
- RED: add tests in crates/mlpl-array/tests/ (or extend an existing ops_test file) for rank-3 concat along axis 2 and rank-4 concat along axis 3. Assert output shape and element-wise content.
- GREEN: drop the 'if axis > 1' branch; rewrite copy_concat_rows to walk the outer dimensions, then copy the per-position 'self' segment followed by the per-position 'other' segment of length (inner_a + inner_b).
- REFACTOR: make sure axis-0 and axis-1 paths still go through the same generalized path (do NOT keep a special-case fast path -- if benchmarks later show a regression we can add one).

Quality gates: cargo test -p mlpl-array; cargo clippy -p mlpl-array --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. No web changes; no pages rebuild.

After this step ships the backward pass still assumes axis < 2 -- step 002 generalizes the tape.