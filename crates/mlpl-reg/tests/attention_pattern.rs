//! Visual regression test for the Attention Pattern demo --
//! last line is `svg(Aself, "heatmap")` where Aself is the
//! softmax of `Qs @ Qs^T / sqrt(4)`, the diagonal-dominant
//! self-attention pattern. 6x6 grid.

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
    "#fde725", "#440154", "#1e1e2e", "#383468", "#258286", "#1e1e2e", "#440154", "#3f175c",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
Qs = randn(17, [6, 4])
Sself = matmul(Qs, transpose(Qs)) / sqrt(4)
Aself = softmax(Sself, 1)
svg(Aself, "heatmap")
"#;

#[test]
fn attention_pattern_self_heatmap_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("attention_pattern", &raster, SAMPLE_POINTS, GOLDEN);
}
