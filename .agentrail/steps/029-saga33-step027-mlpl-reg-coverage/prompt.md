Fill in the visual regression test coverage for the remaining ~28 MLPL demos. Builds on the harness from saga 33 step 026-mlpl-reg-harness.

INPUT: the demo list in apps/mlpl-web/src/demos_basics.rs, demos_models.rs, demos_attention.rs, demos_lm.rs, demos_vit.rs. ~29 demos total; step 026 already covered moons_mlp. Survey which demos actually produce visual output -- a demo that ends in a scalar / array / record produces nothing to test, so it is OUT OF SCOPE for this step.

DELIVERABLE for each in-scope demo:
- A new `crates/mlpl-reg/tests/<demo_slug>.rs` file matching the moons_mlp.rs template from step 026.
- 8-12 (x, y) sample points chosen per the design rules below.
- A hand-verified golden hex array -- generated via `MLPL_REG_PRINT_GOLDEN=1 cargo test ...`, inspected via the /tmp/ diff PNG produced on first run, then pasted into the test file.
- The test must PASS green at commit time. NEVER commit a golden that hasn't been visually verified.

SAMPLE-POINT STRATEGY (per viz type):
- boundary_2d / decision_boundary: 4 corners + 4 mid-edges + 1 center (9 pts). Corners should resolve to the dominant decision class for that region of input space -- this is the structural check that catches "stripes instead of crescents".
- hist / bar / line / loss_curve: 6-8 pts. Peak / first-bar / last-bar / baseline-mid / a few in-between.
- scatter / scatter_labeled / scatter3d: 8 pts. 3-4 known-data-point positions (where a circle should be) + 4 known-empty background positions.
- heatmap / heatmap_grid: 9-10 pts. Corners + center + 2 known-bright cells + 2 known-dark cells.
- confusion_matrix: one pt per cell (2x2 -> 4 pts, 3x3 -> 9). Each cell's color encodes its count.
- gallery / attention_overlay: one pt per tile center, plus a couple of overlay-specific pts (head-marker, axis label).

GOLDEN-CAPTURE WORKFLOW per demo:
1. Run `MLPL_REG_PRINT_GOLDEN=1 cargo test -p mlpl-reg --test <demo_slug> --release -- --nocapture`. The harness writes the actual PNG to /tmp/mlpl-reg-fail/<demo_slug>.png and prints the sampled hexes as a copy-pasteable array.
2. Open the PNG and verify the visual output is what you expect for the demo (e.g. moons crescents, attention diagonal, training loss going down).
3. Paste the printed array into the test file as the GOLDEN const.
4. Re-run without the env flag -- test should now pass.
5. Commit.

BATCHING: do not commit all 28 in one commit. Break into 3-5 commits grouped by viz type:
- Commit A: deterministic charts (hist / bar / line / loss_curve / scatter / scatter3d / scatter_labeled).
- Commit B: classifier surfaces (boundary_2d / decision_boundary -- the saga 33 step 025 fix should be visible here).
- Commit C: heatmaps (heatmap / heatmap_grid / confusion_matrix).
- Commit D: composite renders (gallery / attention_overlay).
Each commit's tests pass green; each commit nets negative on sw-checklist warnings.

OUT OF SCOPE for step 027:
- Adding NEW demos (do those in their own step).
- Changing the harness from step 026 (file a follow-up step if the harness has a gap that prevents a golden from being captured -- DO NOT silently work around it).
- Skipping or xfail'ing a regression test. If a golden cannot be hand-verified (e.g. because the demo output is broken), file a saga-31-style regression step like step 025 and link to it from a TODO comment, but the regression test itself must still be present (failing or pending fix is fine; silently-skipped is not).

DONE = step 027 is done WHEN:
- Every demo that produces a visual output has a corresponding crates/mlpl-reg/tests/<slug>.rs file.
- All regression tests pass green (modulo any TODO-marked pending-fix demos like the moons one IF step 025 is not yet landed).
- The README updated in step 026 lists every demo with its slug + the saga step it tracks.
- sw-checklist net-negative on both axes across the saga 33 paydown trajectory.
