Stand up the visual regression test harness for MLPL demos. This is a NEW test crate (not unit tests in mlpl-viz). Purpose: detect visual drift in demo outputs against a known-good baseline. Companion to the saga 33 step 025-moons-mlp-regression filed earlier -- the harness from this step is what enables the test that step needs.

DESIGN (decided, do not redesign):
- New workspace crate `crates/mlpl-reg/` (or `crates/mlpl-viz-reg/` -- either name is fine, pick one).
- Dev-dependencies: resvg + tiny-skia for SVG -> PNG rasterization in-memory (NO files committed); image crate for pixel access; the eval pipeline (mlpl-eval) to run demos end-to-end.
- Each demo is ONE test file `crates/mlpl-reg/tests/<demo_slug>.rs`.
- Per-test structure (see below) -- two parallel `&[(u32, u32)]` and `&[&str]` constants, plus a one-shot run-demo-and-compare helper from the crate's test-support module.
- Comparison is element-wise on the sampled hex list against the golden hex list. ±3 per-channel RGB tolerance (handles anti-aliasing).
- On failure: write the actual rasterized PNG to `/tmp/mlpl-reg-fail/<demo>.png` and print "see /tmp/mlpl-reg-fail/<demo>.png to inspect"; print which sample(s) mismatched and the delta per channel. NEVER commit the diff PNG.
- A `MLPL_REG_PRINT_GOLDEN=1` env flag re-prints the sampled hexes as a copy-pasteable Rust array (for the bootstrap workflow + when intentionally rebaselining); the regular test path treats the inline `GOLDEN` array as immutable. The "no blind golden rebase" rule in CLAUDE.md still applies: when a test fails, inspect the PNG before deciding the new sampled hexes are correct.

DEMO TEMPLATE (this is the load-bearing artifact for step 026):

```rust
// crates/mlpl-reg/tests/moons_mlp.rs
use mlpl_reg::{run_demo_to_svg, rasterize, sample_hex, MOONS_MLP_DEMO_SOURCE};

const SAMPLE_POINTS: &[(u32, u32)] = &[
    ( 50,  50),  // top-left:     should match upper-moon-class dominant color
    (350,  50),  // top-right:    opposite class
    ( 50, 250),  // bottom-left:  opposite class from top-left
    (350, 250),  // bottom-right: matches upper-moon side again (depends on demo)
    (200, 150),  // center near decision boundary
    (100, 100),
    (300, 100),
    (100, 200),
    (300, 200),
];

const GOLDEN: &[&str] = &[
    "#8bb2f9", "#f28ba8", "#f28ba8", "#8bb2f9",
    "#xxxxxx", "#xxxxxx", "#xxxxxx", "#xxxxxx", "#xxxxxx",
];

#[test]
fn moons_mlp_decision_boundary_matches_baseline() {
    let svg = run_demo_to_svg(MOONS_MLP_DEMO_SOURCE);
    let raster = rasterize(&svg);
    let actual: Vec<String> = SAMPLE_POINTS.iter().map(|(x,y)| sample_hex(&raster, *x, *y)).collect();
    assert_eq!(actual, GOLDEN, "moons_mlp visual regression -- see /tmp/mlpl-reg-fail/moons_mlp.png");
}
```

CRATE CONTENTS for step 026:

- `crates/mlpl-reg/Cargo.toml`: declares deps (mlpl-eval, mlpl-viz, resvg, tiny-skia, image, all dev-deps not main deps -- this crate is tests-only).
- `crates/mlpl-reg/src/lib.rs`: exports `run_demo_to_svg(mlpl_source: &str) -> String`, `rasterize(svg: &str) -> Pixmap`, `sample_hex(raster: &Pixmap, x: u32, y: u32) -> String`. Also exports per-demo source-text consts like `MOONS_MLP_DEMO_SOURCE: &str` so each test file is self-contained (or have each test inline its source -- whatever's cleaner, but the choice has to be uniform across all 29 tests in step 027). Each helper ≤25 LOC; the lib module ≤4 fns.
- `crates/mlpl-reg/tests/moons_mlp.rs`: ONE demo regression test, as above. This step seeds ONLY this one test -- the remaining 28 demos are step 027.
- Workspace `Cargo.toml`: add the new crate as a member.
- `.gitignore`: add `/tmp/mlpl-reg-fail/`.

This step's moons test should FAIL until step 025-moons-mlp-regression lands the fix. That is the test passing its purpose; do NOT skip / xfail it. If you choose to land step 026 BEFORE step 025, mark the failing assertion with a TODO comment pointing to step 025 and let the test fail naturally. Or land 025 first, then 026 -- coordinator's call.

REQUIRED for `mlpl-reg`'s test build:
- Rasterization canvas size MUST match the SVG's intrinsic viewBox (MLPL renders 400x300). Setting a larger rasterization target would re-scale coordinates and invalidate the inline (x, y) tables; pick 400x300 to match the source and document this in the lib doc-comment.
- `run_demo_to_svg` must use the SAME eval entry point that the web playground uses (mlpl-eval's eval_program -> mlpl-viz render path). The whole point is full-stack regression catching -- a renderer-only test is the existing unit-test surface.

QUALITY GATES (saga 33 standard):
1. cargo test -p mlpl-reg --release  must pass for any non-moons golden it ships (only moons in this step).
2. cargo clippy -p mlpl-reg --all-targets --all-features -- -D warnings.
3. cargo fmt --all -- --check.
4. markdown-checker on any docs that changed.
5. sw-checklist must net-negative on FAILs AND warnings vs the previous commit. New crate adds modules to the workspace; since this is a NEW crate (not adding modules to an existing FAILing crate), the new crate must itself pass Crate-Module-Count -- design the helpers to fit in ≤4 modules.
6. Push after commit.

DONE = step 026 is done WHEN:
- mlpl-reg crate exists as a workspace member.
- One regression test (moons_mlp) is wired up and runs to either pass or fail (no skip/xfail).
- README in the crate or a comment in src/lib.rs explains the workflow: how to add a new regression test, how to (re)generate the golden array via MLPL_REG_PRINT_GOLDEN, how to interpret the /tmp/ diff PNG.
- agentrail next will fire 027 to fill in the remaining demo coverage.
