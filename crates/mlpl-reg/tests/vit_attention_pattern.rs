//! Visual regression test for the ViT Attention Pattern
//! (no training) demo -- a 17x17 untrained attention heatmap
//! from a single 64x64 synthetic image's patch tokens + CLS.
//! Test source truncated at `svg(A, "heatmap")`; the demo's
//! trailing `row_sums` sanity check isn't visible viz output.

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
    "#420a58", "#3a2c65", "#1e1e2e", "#440254", "#440154", "#1e1e2e", "#440154", "#420857",
    "#1e1e2e",
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
Wq = randn(501, [128, 128]) / sqrt(128)
Wk = randn(502, [128, 128]) / sqrt(128)
q  = matmul(seq, Wq)
k  = matmul(seq, Wk)
scores = matmul(q, transpose(k)) / sqrt(128)
A = softmax(scores, 1)
svg(A, "heatmap")
"#;

#[test]
fn vit_attention_pattern_heatmap_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("vit_attention_pattern", &raster, SAMPLE_POINTS, GOLDEN);
}
