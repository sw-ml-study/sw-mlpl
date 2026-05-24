//! Visual regression test for the Decoder Block demo --
//! pre-norm causal self-attn + cross-attn over an encoder
//! stand-in. Last viz is the cross-attention weights heatmap.

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
    "#fde725", "#440154", "#1e1e2e", "#fde725", "#440154", "#1e1e2e", "#440154", "#430656",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
T_tgt = 4
T_src = 6
d_model = 8
X_tgt = randn(0, [T_tgt, d_model])
X_src = randn(1, [T_src, d_model])
self_attn = residual(chain(rms_norm(d_model), causal_attention(d_model, 1, 2)))
H = apply(self_attn, X_tgt)
pre_xattn = rms_norm(d_model)
H_norm = apply(pre_xattn, H)
Wq = randn(3, [d_model, d_model])
Wk = randn(4, [d_model, d_model])
Wv = randn(5, [d_model, d_model])
Q = matmul(H_norm, Wq)
K = matmul(X_src, Wk)
V = matmul(X_src, Wv)
weights = softmax(matmul(Q, transpose(K)) / sqrt(d_model), 1)
svg(weights, "heatmap")
"#;

#[test]
fn decoder_block_cross_attention_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("decoder_block", &raster, SAMPLE_POINTS, GOLDEN);
}
