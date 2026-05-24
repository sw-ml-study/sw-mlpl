//! Visual regression test for the Self-Attention from Scratch
//! demo -- a 6-token sequence with Q/K/V projections, scaled
//! softmax attention weights rendered as a 6x6 heatmap.
//! Source truncated at the `svg(weights, "heatmap")` line.

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
    "#440254", "#440154", "#1e1e2e", "#440154", "#fde725", "#1e1e2e", "#440154", "#440154",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
T = 6
d_model = 4
X = randn(0, [T, d_model])
Wq = randn(1, [d_model, d_model])
Wk = randn(2, [d_model, d_model])
Wv = randn(3, [d_model, d_model])
Q = matmul(X, Wq)
K = matmul(X, Wk)
V = matmul(X, Wv)
scores = matmul(Q, transpose(K)) / sqrt(d_model)
weights = softmax(scores, 1)
svg(weights, "heatmap")
"#;

#[test]
fn self_attention_scratch_weights_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("self_attention_scratch", &raster, SAMPLE_POINTS, GOLDEN);
}
