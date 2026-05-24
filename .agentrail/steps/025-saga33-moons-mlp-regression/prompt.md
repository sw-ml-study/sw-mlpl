Investigate and fix the Moons MLP web-demo decision-boundary regression. The `boundary_2d(p1, [30, 30], X, y)` line in `apps/mlpl-web/src/demos_models.rs` ("Moons MLP" demo) renders a vertical-band shading instead of the curved crescent boundary that follows the two scatter clusters. The same regression reproduces unchanged across saga 33 step 023 (current HEAD), pre-step-023, pre-saga-33, AND the commit pages/ was last built from (cfd4348, saga 30 step 005) -- so the regression predates saga 33 by many changes.

Concrete reproducer (script form):
1. Copy the body of the "Moons MLP" demo in apps/mlpl-web/src/demos_models.rs into a .mlpl file (the raw-matmul path, not the model-DSL demos/moons_mlp.mlpl).
2. ./target/release/mlpl-repl -f <script>.mlpl, then inspect the produced SVG.
3. Symptom: rows of the rendered 30x30 grid all share the same left-blue / right-pink color pattern -- p1[r*30+c] varies primarily with c (x-axis) and barely with r (y-axis). Expected: a curved boundary that approximately follows the two interleaved moons in X.

Diagnosis hints:
- demos/moons_mlp.mlpl uses the model DSL (chain + apply) with seeds 11, 12 and DOES train to the curved boundary -- worth comparing why the raw-matmul path with the same seeds does not.
- Likely culprits to bisect (in order of likelihood): moons() generator, randn() seeding, adam optimizer state lifecycle, softmax(_, int_axis) path through eval_fncall -> mlpl_runtime::call_builtin("softmax", ...), matmul label propagation, cross_entropy gradient.
- Try several alternative seeds in the demo (e.g. 7/8, 17/19, 23/29) to confirm whether the failure is seed-specific or systematic. If a small minority of seeds converge correctly, the math is fine and only seed 11/12 lost its old behavior; if all seeds collapse to a near-vertical boundary, an op or grad regressed.

Required deliverable (the saga ratchet still applies for any code-touching commit, but the new test is the load-bearing artifact):

NEW TEST: add a regression test in crates/mlpl-eval/tests/ (e.g. moons_mlp_decision_boundary.rs) that:
- runs the Moons MLP raw-matmul demo end-to-end (lex/parse/eval).
- extracts the post-training `p1` (length-900 grid output).
- asserts that the per-row variance and per-column variance are both above some floor relative to total variance, i.e. the boundary depends on BOTH x and y, not just x. (E.g. row_var > 0.01 * total_var AND col_var > 0.01 * total_var; tune thresholds against a working seed.)
- alternative: assert at least one of (15,0), (15,29), (0,15), (29,15) corners has the expected class, since they're far inside the respective moons.

This is the test we're missing -- the existing all_demos_smoke test only checks that demos run without erroring, not that they produce correct output.

If your fix is a one-line builtin change (e.g. softmax-axis interpretation), the same test should fail before and pass after. If the root cause is a non-deterministic float ordering or a seed-dependent local minimum, the demo source itself may need to switch to seeds that train robustly, OR move to the model DSL like demos/moons_mlp.mlpl.

After the fix:
- rebuild pages/ (apps/mlpl-web changed) -- see CLAUDE.md "Live Demo (GitHub Pages) Deploy".
- commit the test + fix + pages rebuild + .agentrail/ metadata together.
- sw-checklist must net-negative on FAILs and on warnings vs the previous commit.
