//! Visual regression test for the Decision Boundary: logical
//! gates demo. The full demo trains 5 separate logistic
//! regressors (AND, OR, NAND, NOR, XOR) and renders one
//! decision_boundary per. run_demo_to_svg captures the LAST
//! viz, so this test pins XOR's "least-bad linear fit" surface
//! (the pedagogical punchline: roughly uniform 0.5 probability
//! because no line separates XOR).

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
    "#7a9edc", "#7a9edc", "#7a9edc", "#7a9edc", "#7a9edc", "#7a9edc", "#7a9edc", "#7a9edc",
    "#7a9edc",
];

const DEMO_SRC: &str = r#"
X = [[0,0],[0,1],[1,0],[1,1]]
n = 4
lr = 1.0
y = [0, 0, 0, 1]
w = zeros([2])
b = 0
repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }
gx = grid([0, 1, 0, 1], 20)
y = [0, 1, 1, 1]
w = zeros([2])
b = 0
repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }
y = [1, 1, 1, 0]
w = zeros([2])
b = 0
repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }
y = [1, 0, 0, 0]
w = zeros([2])
b = 0
repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }
y = [0, 1, 1, 0]
w = zeros([2])
b = 0
repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }
surface_xor = reshape(sigmoid(reshape(matmul(gx, reshape(w, [2, 1])) + b, [400])), [20, 20])
tp_xor = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
svg(surface_xor, "decision_boundary", tp_xor)
"#;

#[test]
fn decision_boundary_gates_xor_failure_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("decision_boundary_gates", &raster, SAMPLE_POINTS, GOLDEN);
}
