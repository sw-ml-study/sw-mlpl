//! Visual regression test for the Tiny MLP demo --
//! hand-rolled 2-layer MLP (linear + tanh + linear) trained
//! on a 4-blob XOR-style dataset via 600 explicit forward +
//! backward gradient steps. Renders the curved boundary_2d
//! surface over a 30x30 input-space grid.

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
    "#7a4e63", "#93b0f2", "#5a72a0", "#91aff1", "#dc93b9", "#d098c2", "#89b4fa", "#c89bc8",
    "#e584a0",
];

const DEMO_SRC: &str = r#"
D = blobs(3, 20, [[-2,-2],[2,2],[-2,2],[2,-2]])
X = matmul(D, [[1,0],[0,1],[0,0]])
raw = reshape(matmul(D, [[0],[0],[1]]), [80])
y = gt(raw, 1.5)
Y = one_hot(y, 2)
W1 = randn(5, [2, 8]) * 0.5
b1 = zeros([8])
W2 = randn(6, [8, 2]) * 0.5
b2 = zeros([2])
lr = 0.2
repeat 600 { Z1 = matmul(X, W1) + matmul(ones([80, 1]), reshape(b1, [1, 8])); H = tanh_fn(Z1); Z2 = matmul(H, W2) + matmul(ones([80, 1]), reshape(b2, [1, 2])); P = softmax(Z2, 1); dZ2 = P - Y; gW2 = matmul(transpose(H), dZ2) / 80; gb2 = reduce_add(dZ2, 0) / 80; dH = matmul(dZ2, transpose(W2)); dZ1 = dH * (1 - H * H); gW1 = matmul(transpose(X), dZ1) / 80; gb1 = reduce_add(dZ1, 0) / 80; W1 = W1 - lr * gW1; b1 = b1 - lr * gb1; W2 = W2 - lr * gW2; b2 = b2 - lr * gb2 }
G = grid([-4, 4, -4, 4], 30)
GZ1 = matmul(G, W1) + matmul(ones([900, 1]), reshape(b1, [1, 8]))
GH = tanh_fn(GZ1)
GZ2 = matmul(GH, W2) + matmul(ones([900, 1]), reshape(b2, [1, 2]))
GP = softmax(GZ2, 1)
p1 = reshape(matmul(GP, [[0],[1]]), [900])
boundary_2d(p1, [30, 30], X, y)
"#;

#[test]
fn tiny_mlp_decision_boundary_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("tiny_mlp", &raster, SAMPLE_POINTS, GOLDEN);
}
