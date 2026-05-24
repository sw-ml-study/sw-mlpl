//! Visual regression test for the Visualizations web demo --
//! last-line heatmap of `reshape(iota(25), [5, 5])`. The
//! prior svg() calls (scatter, line, bar) are not captured
//! by run_demo_to_svg's last-expression rule; if those need
//! coverage they ship as separate tests.

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
    "#440154", "#3b2562", "#1e1e2e", "#277983", "#339883", "#1e1e2e", "#b4ca47", "#ebe02e",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
svg(reshape(iota(25), [5, 5]), "heatmap")
"#;

#[test]
fn visualizations_heatmap_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("visualizations", &raster, SAMPLE_POINTS, GOLDEN);
}
