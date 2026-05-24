//! Visual regression test for the Analysis Helpers demo --
//! last-line boundary_2d on a synthetic 20x20 gradient surface
//! (linear ramp from 0 to ~1). Two stub training points at
//! (0,0) and (1,1) overlay.

use mlpl_reg::{check_or_print_golden, rasterize, run_demo_to_svg};

const SAMPLE_POINTS: &[(u32, u32)] = &[
    (50, 50),
    (200, 50),
    (350, 50),
    (50, 150),
    (200, 150),
    (350, 150),
    (50, 250),
    (200, 250),
    (350, 250),
];

const GOLDEN: &[&str] = &[
    "#e88eaf", "#eb8dad", "#ed8dac", "#b9a1d4", "#bba0d2", "#bd9fd1", "#8eb1f5", "#90b0f3",
    "#93b0f2",
];

const DEMO_SRC: &str = r#"
boundary_2d(reshape(iota(400), [400]) / 400, [20, 20], [[0,0],[1,1]], [0, 1])
"#;

#[test]
fn analysis_helpers_boundary_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("analysis_helpers", &raster, SAMPLE_POINTS, GOLDEN);
}
