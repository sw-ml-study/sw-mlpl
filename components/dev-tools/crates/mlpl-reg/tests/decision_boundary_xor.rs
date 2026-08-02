//! Visual regression test for the Decision Boundary: XOR
//! (with MLP) demo -- a 2-layer MLP trained on XOR via adam,
//! then rendered as a curved decision_boundary surface with
//! the 4 XOR training points overlaid. Truncated at the
//! svg() call.

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

// Refreshed 2026-08-01 after the E4 step-006 optimizer fix (both
// CPU optimizers now use batched step-start gradients, so the
// 500-step training trajectory legitimately changed). New boundary
// visually inspected: the classic two-band XOR split.
const GOLDEN: &[&str] = &[
    "#d47b96", "#80a2da", "#7a9edc", "#d5bfb7", "#7a9edc", "#8eabd6", "#7a9edc", "#d5a4aa",
    "#d47b96",
];

const DEMO_SRC: &str = r#"
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 1, 1, 0]
mdl = chain(linear(2, 4, 0), tanh_layer(), linear(4, 2, 1))
train 500 { adam(cross_entropy(apply(mdl, X), y), mdl, 0.1, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, X), y) }
gx = grid([0, 1, 0, 1], 20)
logits = apply(mdl, gx)
probs = softmax(logits, 1)
pos = take(transpose(probs), 0, 1)
surface = reshape(pos, [20, 20])
tp = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
svg(surface, "decision_boundary", tp)
"#;

#[test]
fn decision_boundary_xor_mlp_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("decision_boundary_xor", &raster, SAMPLE_POINTS, GOLDEN);
}
