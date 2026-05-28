//! Visual regression test for the Softmax Classifier demo --
//! 3-class linear classifier trained via 300 closed-form
//! softmax+CE gradient steps on 90 blob points, rendered as
//! a three-wedge boundary_2d surface.

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
    "#617cad", "#89b3f9", "#6d475c", "#8ab3f8", "#d895bc", "#8ab3f8", "#e490b3", "#5a72a0",
    "#e092b6",
];

const DEMO_SRC: &str = r#"
D = blobs(11, 30, [[0, 0], [4, 4], [-4, 4]])
X = matmul(D, [[1,0],[0,1],[0,0]])
tl = reshape(matmul(D, [[0],[0],[1]]), [90])
Y = one_hot(tl, 3)
W = zeros([2, 3])
b = zeros([3])
lr = 0.2
repeat 300 { logits = matmul(X, W) + matmul(ones([90, 1]), reshape(b, [1, 3])); P = softmax(logits, 1); dZ = P - Y; gW = matmul(transpose(X), dZ) / 90; gb = reduce_add(dZ, 0) / 90; W = W - lr * gW; b = b - lr * gb }
G = grid([-6, 6, -3, 6], 30)
gl = matmul(G, W) + matmul(ones([900, 1]), reshape(b, [1, 3]))
gP = softmax(gl, 1)
p0 = reshape(matmul(gP, [[1],[0],[0]]), [900])
boundary_2d(p0, [30, 30], X, tl)
"#;

#[test]
fn softmax_classifier_boundary_matches_baseline() {
    let svg = run_demo_to_svg(DEMO_SRC).expect("run_demo_to_svg");
    let raster = rasterize(&svg).expect("rasterize");
    check_or_print_golden("softmax_classifier", &raster, SAMPLE_POINTS, GOLDEN);
}
