//! Visual regression test for the Tiny LM Generate demo --
//! trains a 1-layer transformer LM for 30 adam steps, then
//! extracts attention_weights on the prompt "the quick"
//! and renders the [T, T] attention heatmap.

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
    "#fde725", "#440154", "#1e1e2e", "#334970", "#440154", "#1e1e2e", "#3a2a64", "#383267",
    "#1e1e2e",
];

const DEMO_SRC: &str = r#"
corpus = load_preloaded("tiny_corpus")
tok    = train_bpe(corpus, 260, 0)
ids    = apply_tokenizer(tok, corpus)
X_all = shift_pairs_x(ids, 8)
Y_all = shift_pairs_y(ids, 8)
X     = reshape(X_all, [reduce_mul(shape(X_all))])
Y     = reshape(Y_all, [reduce_mul(shape(Y_all))])
V = 260 ; d = 16 ; h = 1
model = chain(embed(V, d, 0), causal_attention(d, h, 1), rms_norm(d), linear(d, V, 2))
experiment "tiny_lm_gen" { train 30 { adam(cross_entropy(apply(model, X), Y), model, 0.01, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(model, X), Y) } }
viz_ids = apply_tokenizer(tok, "the quick")
attn_w  = attention_weights(model, viz_ids)
svg(attn_w, "heatmap")
"#;

#[test]
fn tiny_lm_generate_attention_heatmap_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("tiny_lm_generate", &raster, SAMPLE_POINTS, GOLDEN);
}
