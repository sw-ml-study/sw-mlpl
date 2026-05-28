//! Visual regression test for the Encoder Block demo --
//! pre-norm self-attn + residual + pre-norm FFN + residual,
//! ending with a 4x4 attention_weights heatmap.

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
    "#440154", "#440154", "#1e1e2e", "#440154", "#430556", "#1e1e2e", "#6daf68", "#3b2662",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
X = randn(0, [4, 8])
bare_attn = attention(8, 1, 1)
svg(attention_weights(bare_attn, X), "heatmap")
"#;

#[test]
fn encoder_block_attention_weights_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("encoder_block", &raster, SAMPLE_POINTS, GOLDEN);
}
