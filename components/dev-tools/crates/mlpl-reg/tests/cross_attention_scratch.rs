//! Visual regression test for the Cross-Attention from
//! Scratch demo -- a non-square [T_tgt=4, T_src=6] attention
//! pattern. Source truncated at `svg(weights, "heatmap")`.

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
    "#440154", "#440154", "#1e1e2e", "#440154", "#440154", "#1e1e2e", "#fde725", "#440154",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
T_tgt = 4
T_src = 6
d_model = 4
X_tgt = randn(0, [T_tgt, d_model])
X_src = randn(1, [T_src, d_model])
Wq = randn(2, [d_model, d_model])
Wk = randn(3, [d_model, d_model])
Wv = randn(4, [d_model, d_model])
Q = matmul(X_tgt, Wq)
K = matmul(X_src, Wk)
V = matmul(X_src, Wv)
scores = matmul(Q, transpose(K)) / sqrt(d_model)
weights = softmax(scores, 1)
svg(weights, "heatmap")
"#;

#[test]
fn cross_attention_scratch_weights_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("cross_attention_scratch", &raster, SAMPLE_POINTS, GOLDEN);
}
