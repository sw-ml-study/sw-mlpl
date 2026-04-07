//! Integration test for the PCA demo: power iteration recovers the
//! dominant direction of a linearly-mixed gaussian dataset.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

#[test]
fn pca_power_iteration_recovers_mixing_direction() {
    // Xraw ~ N(0, I)_60x2, then X = Xraw * [[1, 2], [0, 0.3]].
    // The dominant row direction of the mixing matrix (and thus the
    // principal axis of X) is proportional to (1, 2) / sqrt(5).
    let src = r#"
Xraw = randn(1, [60, 2])
X = matmul(Xraw, [[1, 2],[0, 0.3]])
cm = reduce_add(X, 0) / 60
Xc = X - matmul(ones([60, 1]), reshape(cm, [1, 2]))
Cov = matmul(transpose(Xc), Xc) / 60
v = [1, 0]
repeat 20 {
  v = matmul(Cov, v);
  v = v / sqrt(dot(v, v))
}
v
"#;
    let v = eval(src);
    assert_eq!(v.data().len(), 2);
    let (vx, vy) = (v.data()[0], v.data()[1]);
    // True direction (1, 2) / sqrt(5).
    let inv = 1.0 / 5.0f64.sqrt();
    let cos = (vx * inv + vy * 2.0 * inv).abs();
    assert!(cos > 0.95, "|cos| = {cos}, v = ({vx}, {vy})");
}
