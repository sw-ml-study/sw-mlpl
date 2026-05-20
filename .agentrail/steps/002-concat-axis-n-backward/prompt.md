Tier 1 saga step 002: extend the autograd tape's concat backward to split the upstream gradient along any axis.

Today crates/mlpl-eval/src/grad.rs:164 handles the concat backward but the implementation matches the rank/axis assumptions of the pre-step-001 forward (axis < 2). With step 001 shipped, the forward accepts arbitrary axis; the backward now needs to follow.

The concat backward semantics: if forward took two inputs of shapes shapeA and shapeB and concat'd along axis, the upstream gradient comes back with output shape and must be split into two slabs along the same axis, one matching shapeA and one matching shapeB. Route each slab back to its corresponding parent tape node.

TDD (Red/Green/Refactor):
- RED: add a finite-difference gradcheck test in crates/mlpl-eval/tests/ (or extend an existing gradcheck file) on a rank-3 concat-along-axis-2 fixture. Compare analytic gradient (via grad()) against numerical gradient (perturb each input element by epsilon=1e-3, observe loss delta, divide). Tolerance: 1e-3 absolute for fp32.
- GREEN: implement the general-axis split in the backward dispatch.
- REFACTOR: extract a split_along_axis helper if the logic gets long; keep grad.rs's per-op functions short.

Quality gates: cargo test -p mlpl-eval; cargo clippy -p mlpl-eval --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. No web changes.