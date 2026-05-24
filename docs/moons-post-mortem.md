# Post-Mortem: Moons MLP Decision-Boundary Y-Flip

Status: closed. Fix shipped in `c8790f5` (saga 33 step 025) on 2026-05-24.
Author: written in response to user request after the fix landed.

## TL;DR

The decision-boundary surface rendered by `boundary_2d` and
`decision_boundary` was vertically mirrored relative to its own
overlaid training points, making the colored shading appear
"out of phase" with the scatter dots on every demo that paired
the two (most visibly the Moons MLP demo). The bug was a
1-line indexing mismatch between two helpers that should have
agreed on which way `y` points; it had been latent in the
codebase for **47 days** (introduced 2026-04-07, fixed
2026-05-24). The fix is 1 line per renderer plus a small
helper extraction. The bug escaped four layers of pre-existing
test infrastructure because none of those layers actually
looked at the visual output as a 2-D picture.

## Symptom

The user reported, on 2026-05-24, that the Moons MLP demo's
`boundary_2d(p1, [30, 30], X, y)` line in the web playground
rendered colored shading whose curves were offset from the
scatter dots that should sit inside them. Their summary: "the
shaded area colors no longer match the scatter graph. They
used to make the same crescent shape but now appear out of
phase."

The user's framing was that the demo had *regressed* recently.
Bisect proved it had not -- the same misalignment reproduced
unchanged at four historical points (current HEAD, pre-step-023,
pre-saga-33, and `cfd4348` -- the commit the live demo at
github.io was last built from). The bug had been there since
the renderers were first added, but only became visually salient
on a demo (Moons MLP) where the user could overlay the colored
shading on top of the dots and notice that they didn't match.

## Root Cause

Two SVG renderers in `mlpl-viz` -- `analysis_boundary_2d` in
`crates/mlpl-viz/src/analysis.rs` and `draw_surface` in
`crates/mlpl-viz/src/svg/decision_boundary.rs` -- iterated their
input grid as `raw[r * cols + c]` with `r=0` placed at image
top. Both renderers immediately call `draw_labeled_points` or
`draw_points` to overlay scatter dots, and *those* helpers use
the standard math-y-bottom convention via `cy = H - PAD - ty *
plot_h`. The two conventions disagreed, so the rendered surface
was the vertical mirror image of its own overlaid points.

The fix is to read the grid with `r` flipped:

```diff
- let t = ((raw[r * cols + c] - lo) / span).clamp(0.0, 1.0);
+ let t = ((raw[(rows - 1 - r) * cols + c] - lo) / span).clamp(0.0, 1.0);
```

That single-character change in each renderer makes the surface
agree with its own overlay. (See `c8790f5` for the full
commit, which also lifts input validation into
`crates/mlpl-viz/src/boundary_2d_validate.rs` to keep
`analysis_boundary_2d` under the 25-LOC function-LOC budget.)

The bug was latent for so long because:

- *Either* helper in isolation has a perfectly self-consistent
  y convention. The bug only appears in the *interaction* between
  them.
- The Moons MLP demo is the only common demo whose surface and
  dots should visually coincide along the y-axis. Demos that
  paired the surface with a uniform background (e.g. the
  five-gate Decision Boundary demo seeded in `8557153`) hid the
  flip behind a featureless overlay.
- The bug LOOKS like an under-trained model rather than a
  rendering bug. A flipped surface still has a smooth, plausible
  boundary; the user's natural conclusion is "model didn't
  converge for this seed", not "renderer is broken".

## Timeline

| Date       | Commit    | Event                                                                                 |
|------------|-----------|----------------------------------------------------------------------------------------|
| 2026-04-07 | `1d08bfd` | `decision_boundary` renderer introduced. Y-flip bug present from this commit.          |
| 2026-04-07 | `a038e5d` | `boundary_2d` (sibling renderer) introduced with the same y-flip bug, ~80 min later.   |
| 2026-04-11 | `11e1515` | `reg-rs` regression suite seeded for 31 demos. **Baselines captured the buggy SVG.**   |
| 2026-04-12 | `cfd4348` | Last `pages/` rebuild for the live demo. From this point on the github.io site has the bug. |
| 2026-05-20 | `8557153` | Decision Boundary demo expanded to five logic gates -- but with featureless overlays that don't expose the flip. |
| 2026-05-24 | (user)    | User notices the misalignment on the Moons MLP demo and reports it.                    |
| 2026-05-24 | `c8790f5` | Fix lands: 1-line flip per renderer + 2 new regression tests + 1 e2e test.             |

Total latency: **47 days** (~7 weeks).

## Defect-Escape Analysis

The codebase had four layers of testing for visual outputs when
the bug was introduced. None of them caught it. This is the
core of the post-mortem.

### Layer 1: structural unit tests in `crates/mlpl-viz/tests/`

What they check: the rendered SVG starts with `<svg`, ends with
`</svg>`, contains the expected NUMBER of `<rect>` and `<circle>`
elements (e.g. `assert_eq!(svg.matches("<circle").count(), 2)`),
and includes legend labels for given min/max values.

What they don't check: where any element sits, what color it is,
or whether the layout matches the input data.

Why the bug escaped: a vertically-flipped surface emits the
exact same set of structural elements -- same rect count, same
fill colors, same legend text -- just with the rows in reverse.
Every structural assertion passes whether the flip is present
or not.

### Layer 2: `all_demos_smoke` in `crates/mlpl-eval/tests/`

What it checks: every web playground demo lexes, parses, and
evaluates without throwing an error.

What it doesn't check: what the demo's output values *are*. A
demo that produces a correct SVG string and a demo that produces
a completely wrong SVG string both PASS this test as long as
neither one errors.

Why the bug escaped: the renderer returned a perfectly valid
SVG string. The test only cares about the absence of `Err`.

### Layer 3: `reg-rs` regression suite seeded in `11e1515`

What it does: runs `mlpl-repl -f <demo>.mlpl`, captures stdout
through `normalize-mlpl-output.sh`, and asserts byte-equality
against a baseline `.rgt` file.

What it doesn't check (this is the load-bearing failure mode):
the normalize filter explicitly collapses SVG output to
`[svg: NNNN bytes]` -- the byte count is wildcarded with the
comment "byte count can shift by a digit if the SVG output
format changes in a semantically-equivalent way". So the
captured baseline records *that a viz was produced*, not *what
the viz looks like*.

The deeper problem: even if `reg-rs` had captured the full SVG
text, the baseline was seeded on 2026-04-11 -- four days AFTER
the bug was introduced. The reference was already wrong. A
"regression suite" can only catch deviations from baseline;
it cannot tell you the baseline is incorrect. **Capturing the
output of a buggy system as the golden reference is the way
regression suites silently calcify bugs.**

### Layer 4: human eyeball at demo authoring time

What it should have caught: the original author of the moons /
classifier demos staring at the rendered SVG and noticing the
shading doesn't match the dots.

Why it didn't: best guess from the demo evolution -- when the
renderers were first added the authoring focus was on producing
*some* curved surface that varied with the input grid. The
crescent-shape match between shading and dots is a higher-order
quality check that only matters once you have a working model
+ working renderer + working data pairing. By the time those
three lined up, the demo was already "shipping correctly" by
all four lower layers. The flip looks like an under-trained
model, and "this seed picks a bad local minimum" is a much more
common failure mode than "the renderer is vertically inverted".

### Summary: every layer was about presence, not correctness

| Layer            | Asserts                                            | What it misses                                                    |
|------------------|----------------------------------------------------|--------------------------------------------------------------------|
| Unit (mlpl-viz)  | SVG well-formed, element counts match              | Element *positions* and *colors*                                   |
| Smoke            | Demos don't panic                                  | Output values are correct                                          |
| reg-rs           | stdout matches a normalized baseline               | SVG content (wildcarded); baseline correctness                     |
| Human review     | Visual match between shading and dots              | Did not happen for the boundary renderers                           |

The bug was perfectly positioned to slip through ALL of them.

## Would Current Tests Catch a Reintroduction?

After the fix landed in `c8790f5`, the test surface for this
specific bug class is:

1. **`crates/mlpl-viz/tests/svg_boundary_orientation_tests.rs`**
   (new in step 025). Builds a synthetic row-indexed grid where
   row 0 maps to `t=0` (blue ramp) and row N-1 maps to `t=1`
   (pink ramp). Renders the SVG, parses out every plot-area
   `<rect>`, sorts by image-y, and asserts the top-of-image rect
   has a HIGHER red channel than the bottom-of-image rect.

   This test is the categorical defense against this bug class:
   it fixes the renderer's y-orientation invariant independently
   of any specific model, dataset, seed, or training procedure.
   Verified to FAIL on the pre-fix code and PASS on the post-fix
   code. Reintroducing the bug -- whether by reverting the
   1-line fix, swapping rows during a refactor, or accidentally
   flipping `r` again in some new helper -- causes this test
   to fail loudly with a message that names the saga step.

2. **`crates/mlpl-eval/tests/moons_mlp_decision_boundary.rs`**
   (new in step 025). Runs the live Moons MLP demo from
   `apps/mlpl-web/src/demos_models.rs` end-to-end and asserts
   the post-training `p1` grid has substantial variance along
   BOTH axes (rules out vertical-stripe collapse) and that four
   corners resolve to expected classes.

   This catches a different and broader bug class: the underlying
   model degrading to a single-axis decision rule for reasons
   unrelated to the renderer (a gradient bug, a softmax-axis
   bug, an init bug). It would also catch the y-flip bug
   transitively if the flip somehow propagated back into the
   eval pipeline.

3. **`crates/mlpl-reg/`** (filed as saga 33 step 026, not yet
   landed). The full visual-regression harness will rasterize
   each demo's SVG via `resvg` and sample 8-12 known pixel
   positions per demo, comparing the sampled hex colors against
   a hand-verified inline array. This is the categorical
   defense for *any* renderer drift on *any* demo.

The combination of (1) and (3) means a reintroduction of the
exact y-flip bug would be caught by the unit test in
milliseconds (no eval needed). A novel renderer bug -- different
mechanism, same symptom class -- would be caught by (3) once
the harness has goldens for every demo.

What we *still* cannot catch automatically: a demo whose visual
output is "plausible but wrong" the very first time a golden is
captured. This is the same failure mode that calcified the
y-flip in `reg-rs`. The harness mitigates it by requiring a
manual visual inspection step before any golden is committed
(documented in the step 026 prompt as the bootstrap workflow),
but ultimately a wrong-from-day-one bug requires either a
mathematical invariant (like the orientation test) or a human
who knows what the output should look like.

## Why It Took So Long to Notice

Several factors stretched the latency from "minutes after demo
author looks at it" to "47 days":

- **The bug looks like an under-trained model.** A flipped
  surface still has a smooth, sensible curve; the wrong-place
  curve looks like the kind of fit you get from a random seed
  that landed in a bad local minimum. "Try a different seed"
  is the conventional response, not "audit the renderer".
- **All four test layers passed.** Without any automated
  signal, the bug only surfaces from a user looking at the
  picture and knowing what it *should* look like.
- **The Moons MLP demo is the demo most exposed to this bug.**
  The other classifier demos use uniform-background datasets
  (Decision Boundary's logic gates) or fitted hyperplanes (PCA,
  k-means) that don't have the same crescent-overlay structure.
  Most users probably never opened Moons MLP.
- **Pages/ shipped the buggy SVG on 2026-04-12** and was not
  rebuilt since. Whatever fraction of users hit the live demo
  saw the bug, but the github.io page is not heavily trafficked
  enough to surface defect reports quickly.

## Lessons

1. **Structural assertions are necessary but not sufficient
   for rendered output.** Counting elements, validating
   well-formedness, and pattern-matching legend text catches
   gross renderer breakage but misses the entire class of
   "shapes are right, positions are wrong". The new
   `boundary_2d_renders_grid_row_0_at_image_bottom` test
   demonstrates the minimum viable orientation assertion --
   one sample at the top of the image, one at the bottom, plus
   a relation between them. Every renderer that pairs surfaces
   with overlays should have a similar test.

2. **Don't seed a regression baseline before a human has
   confirmed it's correct.** The `reg-rs` baseline captured
   the bug as the golden reference. This is structurally
   indistinguishable from "no test at all" plus a confidence-
   boosting checkmark that gets in the way of recognizing the
   gap. Saga 33 step 026's bootstrap workflow makes the manual
   inspection step explicit and load-bearing: the user runs
   with `MLPL_REG_PRINT_GOLDEN=1`, opens the generated PNG,
   confirms it visually, and *then* pastes the hex array into
   the test source. Goldens captured silently from a buggy
   system are worse than no golden at all.

3. **Two helpers in the same crate that share a coordinate
   convention should reference a SINGLE shared definition of
   that convention.** Both renderers in this bug independently
   chose a y direction; they happened to disagree. A shared
   `coords` module with a single `image_y_for_row(r, rows)`
   helper would have forced the disagreement to surface at
   review time. (Not done in step 025's fix -- the two
   renderers still implement the flip independently. A
   follow-up consolidation could lift this into a shared
   helper if the same bug class shows up a third time.)

4. **Bisect first, then accuse.** The user's initial report
   framed this as a recent regression. Two bisect runs
   (against `f0528f4` pre-step-023 and `87396e8` pre-saga-33)
   proved the SVG was byte-identical at those points -- so the
   real investigation could skip "what did saga 33 break?"
   entirely and look at the renderer math directly. This
   saved hours of staring at the wrong code.

## Action Items

- [x] Fix the renderer y-flip in both `boundary_2d` and
      `decision_boundary` (shipped in `c8790f5`).
- [x] Add the orientation regression test
      (`svg_boundary_orientation_tests.rs`).
- [x] Add the end-to-end Moons MLP regression test
      (`moons_mlp_decision_boundary.rs`).
- [ ] **Land saga 33 step 026** (`mlpl-reg` harness) to give
      every viz a hex-sample regression test, replacing the
      stdout-wildcarded `reg-rs` suite for visual content
      assertions.
- [ ] **Land saga 33 step 027** (`mlpl-reg` coverage) to
      hand-verify a golden for each of the ~29 web playground
      demos.
- [ ] Rebuild `pages/` to deploy the fix to the live demo at
      `https://sw-ml-study.github.io/sw-mlpl/`. The current
      build there has shipped the buggy boundary since
      2026-04-12.
- [ ] Consider consolidating the row-index flip into a shared
      `coords` helper if a third renderer in this family is
      added. Not strictly necessary today; flag for review next
      time the boundary-render code is touched.
- [ ] Audit the rest of the `reg-rs` baselines for the same
      "captured a buggy output as golden" failure mode. The
      `normalize-mlpl-output.sh` wildcarding means many demos'
      SVG content is invisible to `reg-rs`; the new
      `mlpl-reg` harness should supersede those entries rather
      than just sit alongside them.
