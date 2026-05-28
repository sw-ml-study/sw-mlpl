//! Visual regression test for the Moons MLP web demo.
//!
//! Mirrors the demo source from
//! `apps/mlpl-web/src/demos_models.rs`. Sampled hex colors
//! were captured with `MLPL_REG_PRINT_GOLDEN=1` after saga 33
//! step 025 fixed the boundary y-flip, and visually verified
//! against the rendered PNG. If this test fails after a
//! legitimate change (different model, different demo
//! source, intentional renderer tweak), re-run with
//! `MLPL_REG_PRINT_GOLDEN=1`, INSPECT the PNG at
//! `/tmp/mlpl-reg-fail/moons_mlp.png`, and only then update
//! the `GOLDEN` array below.

use mlpl_reg::{check_or_print_golden, rasterize, run_demo_to_svg};

/// Image-space sample points on the 400x300 boundary canvas.
/// `PAD=30`, so plot area is x in [30, 370], y in [30, 270].
/// 9 points: 4 interior corners, 4 mid-edges, 1 center.
const SAMPLE_POINTS: &[(u32, u32)] = &[
    (50, 50),   // image TL: upper-left of plot area
    (200, 50),  // top mid
    (350, 50),  // image TR
    (50, 150),  // left mid
    (200, 150), // center
    (350, 150), // right mid
    (50, 250),  // image BL
    (200, 250), // bottom mid
    (350, 250), // image BR
];

/// Hex colors captured 2026-05-24 after the saga 33 step 025
/// boundary y-flip fix landed. Verified visually against the
/// rendered PNG: blue dominant in the upper portion (matches
/// upper-crescent class 0 dots), pink dominant in the lower
/// portion (matches lower-crescent class 1 dots), curved
/// transition through the middle.
const GOLDEN: &[&str] = &[
    "#89b3f9", "#82aaed", "#89b3f9", "#89b3f9", "#89b3f9", "#e484a0", "#8cb2f7", "#f28ba8",
    "#f28ba8",
];

const DEMO_SRC: &str = r#"
M = moons(7, 120, 0.08)
X = matmul(M, [[1,0],[0,1],[0,0]])
y = reshape(matmul(M, [[0],[0],[1]]), [120])
O120 = ones([120, 1])
W1 = param[2, 8]
b1 = param[1, 8]
W2 = param[8, 2]
b2 = param[1, 2]
W1 = randn(11, [2, 8]) * 0.5
W2 = randn(12, [8, 2]) * 0.5
train 200 { adam(cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y) }
G = grid([-1.5, 2.5, -1, 1.5], 30)
O900 = ones([900, 1])
GZ1 = matmul(G, W1) + matmul(O900, b1)
GH = tanh_fn(GZ1)
GZ2 = matmul(GH, W2) + matmul(O900, b2)
GP = softmax(GZ2, 1)
p1 = reshape(matmul(GP, [[0],[1]]), [900])
boundary_2d(p1, [30, 30], X, y)
"#;

#[test]
fn moons_mlp_decision_boundary_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("moons_mlp", &raster, SAMPLE_POINTS, GOLDEN);
}
