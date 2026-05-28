//! Visual regression test for the ViT Multi-Head Attention
//! Pattern (no training) demo -- 4 untrained heads rendered as
//! a 2x2 heatmap_grid. Source truncated at
//! `svg(A, "heatmap_grid")`.

use mlpl_reg::{check_or_print_golden, rasterize, run_demo_to_svg};

const SAMPLE_POINTS: &[(u32, u32)] = &[
    (75, 75),   // top-left tile interior
    (275, 75),  // top-right tile interior
    (75, 200),  // bottom-left tile interior
    (275, 200), // bottom-right tile interior
    (50, 50),
    (200, 50),
    (50, 250),
    (200, 250),
    (350, 250),
];

const GOLDEN: &[&str] = &[
    "#440154", "#440154", "#998339", "#440154", "#440154", "#fde725", "#440154", "#440154",
    "#e0dc33",
];

const DEMO_SRC: &str = r#"
img = randn(101, [1, 3, 64, 64])
tokens_4d = patchify(img, 16)
tokens    = reshape(tokens_4d, [16, 768])
Wp        = randn(201, [768, 128]) / sqrt(768)
patches   = matmul(tokens, Wp)
cls = randn(301, [1, 128])
seq_no_pos = concat(cls, patches, 0)
pos = randn(401, [17, 128])
seq = seq_no_pos + pos
mdl = attention(128, 4, 17)
A = attention_weights(mdl, seq)
svg(A, "heatmap_grid")
"#;

#[test]
fn vit_multihead_attention_pattern_grid_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden(
        "vit_multihead_attention_pattern",
        &raster,
        SAMPLE_POINTS,
        GOLDEN,
    );
}
