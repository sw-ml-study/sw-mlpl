//! Integration tests for the k-means Lloyd iteration expressed in MLPL.
//!
//! These tests exercise the same vectorized formulation used by
//! `demos/kmeans.mlpl` so that a regression in built-ins or evaluator
//! broadcasting will show up here before the demo breaks.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

#[test]
fn kmeans_two_clusters_converges() {
    // 6 points: three near (0, 0), three near (5, 5).
    // Initial centers deliberately placed near each true cluster so
    // the assignment is stable after one iteration.
    let src = r#"
X = [[0,0],[0.2,-0.1],[-0.1,0.1],[5,5],[5.1,4.9],[4.9,5.2]]
C = [[0.5,0.5],[4.5,4.5]]
N = 6
K = 2
repeat 5 {
  sqX = reshape(reduce_add(X*X, 1), [6,1]);
  sqC = reshape(reduce_add(C*C, 1), [1,2]);
  XC  = matmul(X, transpose(C));
  dists = matmul(sqX, ones([1,2])) + matmul(ones([6,1]), sqC) - 2*XC;
  clus  = argmax(-1 * dists, 1);
  jj = reshape(iota(2), [2,1]);
  ll = reshape(clus, [1,6]);
  diff = matmul(jj, ones([1,6])) - matmul(ones([2,1]), ll);
  A = eq(diff, 0);
  counts = reshape(reduce_add(A, 1), [2,1]);
  sums   = matmul(A, X);
  C = sums / matmul(counts, ones([1,2]))
}
C
"#;
    let c = eval(src);
    assert_eq!(c.shape().dims(), &[2, 2]);
    let d = c.data();
    // Two centers should be near the two true cluster means (order
    // preserved since initial guess was already in the right basin).
    assert!(d[0].abs() < 0.3 && d[1].abs() < 0.3, "c0={:?}", &d[..2]);
    assert!(
        (d[2] - 5.0).abs() < 0.3 && (d[3] - 5.0).abs() < 0.3,
        "c1={:?}",
        &d[2..]
    );
}

#[test]
fn kmeans_assigns_blobs_points_to_their_true_clusters() {
    // Use blobs() to generate a 3-cluster dataset and run one step of
    // Lloyd assignment with the *true* centers; every point should be
    // assigned to its own label.
    let src = r#"
D = blobs(7, 20, [[0,0],[5,5],[-5,5]])
# extract the 60x2 point matrix and the label vector
X  = matmul(D, [[1,0],[0,1],[0,0]])
tl = reshape(matmul(D, [[0],[0],[1]]), [60])
C  = [[0,0],[5,5],[-5,5]]
sqX = reshape(reduce_add(X*X, 1), [60,1])
sqC = reshape(reduce_add(C*C, 1), [1,3])
XC  = matmul(X, transpose(C))
dists = matmul(sqX, ones([1,3])) + matmul(ones([60,1]), sqC) - 2*XC
clus = argmax(-1 * dists, 1)
mean(eq(clus, tl))
"#;
    let acc = eval(src).data()[0];
    assert!(acc > 0.95, "assignment accuracy {acc} too low");
}
