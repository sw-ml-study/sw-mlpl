//! Saga 33 step 025: end-to-end regression test for the Moons MLP
//! web demo's decision boundary.
//!
//! Runs the raw-matmul + cross_entropy training body that lives in
//! `apps/mlpl-web/src/demos_models.rs` ("Moons MLP" demo), captures
//! the resulting 30x30 `p1` grid (per-grid-point class-1
//! probability), and asserts:
//!
//! 1. The grid varies along BOTH axes (model didn't collapse to a
//!    vertical or horizontal line).
//! 2. Specific corners far from both moons resolve to a sensible
//!    class given seed-11/12 training.
//!
//! Pairs with `crates/mlpl-viz/tests/svg_boundary_orientation_tests.rs`,
//! which catches the renderer-side y-flip bug that step 025 also
//! fixed. The renderer test asserts orientation; this test asserts
//! the underlying p1 grid is non-degenerate.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

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
p1
"#;

fn run_demo() -> Vec<f64> {
    let tokens = lex(DEMO_SRC).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::default();
    match eval_program_value(&stmts, &mut env).expect("eval") {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("expected Value::Array, got {other:?}"),
    }
}

fn axis_mean_vars(p1: &[f64], n: usize) -> (f64, f64, f64, f64) {
    let total = p1.iter().sum::<f64>() / (n * n) as f64;
    let total_var = p1.iter().map(|v| (v - total).powi(2)).sum::<f64>() / (n * n) as f64;
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    for r in 0..n {
        for c in 0..n {
            row_means[r] += p1[r * n + c];
            col_means[c] += p1[r * n + c];
        }
    }
    for v in row_means.iter_mut() {
        *v /= n as f64;
    }
    for v in col_means.iter_mut() {
        *v /= n as f64;
    }
    let row_var = row_means.iter().map(|m| (m - total).powi(2)).sum::<f64>() / n as f64;
    let col_var = col_means.iter().map(|m| (m - total).powi(2)).sum::<f64>() / n as f64;
    (total, total_var, row_var, col_var)
}

#[test]
fn moons_mlp_p1_varies_along_both_axes() {
    let p1 = run_demo();
    assert_eq!(p1.len(), 900);
    let (mean, total_var, row_var, col_var) = axis_mean_vars(&p1, 30);
    assert!(
        total_var > 0.05,
        "expected non-trivial total variance, got {total_var:.4} (mean={mean:.4})"
    );
    // Both row-mean and col-mean variance must be a meaningful
    // fraction of the total. With seed 11/12 the actual numbers
    // are ~46% and ~27%; we set the floor at 5% to leave room for
    // future model / init / op tweaks but still catch any
    // collapse into a single-axis decision rule.
    let row_frac = row_var / total_var;
    let col_frac = col_var / total_var;
    assert!(
        row_frac > 0.05,
        "row variance {row_var:.4} is too small relative to total {total_var:.4} \
         ({:.0}%) -- model may have collapsed to a horizontal cut",
        row_frac * 100.0
    );
    assert!(
        col_frac > 0.05,
        "col variance {col_var:.4} is too small relative to total {total_var:.4} \
         ({:.0}%) -- model may have collapsed to a vertical cut",
        col_frac * 100.0
    );
}

#[test]
fn moons_mlp_corners_resolve_to_expected_classes() {
    let p1 = run_demo();
    // Corner positions in the 30x30 grid (r, c) along with the
    // class the trained model is expected to produce given
    // seed-11/12 deterministic training. These corners are far
    // from any moon dot so they reflect the model's extrapolation
    // rule, not training accuracy. The pinned values come from
    // the analysis run with the fix applied -- update them if
    // the model / training hyperparameters intentionally change.
    let corner = |r: usize, c: usize| -> f64 { p1[r * 30 + c] };
    let tl = corner(0, 0); // math (xmin, ymin) -- below+left of both moons
    let tr = corner(0, 29); // math (xmax, ymin) -- below+right of lower moon
    let bl = corner(29, 0); // math (xmin, ymax) -- above+left of upper moon
    let br = corner(29, 29); // math (xmax, ymax) -- above+right of both moons
    // tl/bl/br all lie above or to the left of the lower-moon
    // extent; the trained classifier should pick class 0 (p1<0.5).
    // tr lies right under the lower-moon's right tail; class 1
    // (p1>0.5).
    assert!(tl < 0.5, "expected tl<0.5 (class 0), got p1={tl:.4}");
    assert!(tr > 0.5, "expected tr>0.5 (class 1), got p1={tr:.4}");
    assert!(bl < 0.5, "expected bl<0.5 (class 0), got p1={bl:.4}");
    assert!(br < 0.5, "expected br<0.5 (class 0), got p1={br:.4}");
}
