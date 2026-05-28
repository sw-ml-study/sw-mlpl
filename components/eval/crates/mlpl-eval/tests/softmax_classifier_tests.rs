//! Integration test: linear softmax classifier on a blobs() dataset
//! should reach >95% accuracy after a few hundred gradient steps.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

#[test]
fn softmax_classifier_reaches_high_accuracy_on_blobs() {
    // 3 centers, 30 points per class -> N = 90, K = 3, D = 2.
    // Note: blobs uses sigma 0.15, so centers spaced by ~4 are very
    // well separated and a linear classifier should nail it.
    let src = r#"
D = blobs(11, 30, [[0, 0], [4, 4], [-4, 4]])
X = matmul(D, [[1,0],[0,1],[0,0]])
tl = reshape(matmul(D, [[0],[0],[1]]), [90])
Y = one_hot(tl, 3)
W = zeros([2, 3])
b = zeros([3])
lr = 0.2
repeat 300 {
  logits = matmul(X, W) + matmul(ones([90, 1]), reshape(b, [1, 3]));
  P = softmax(logits, 1);
  dZ = P - Y;
  gW = matmul(transpose(X), dZ) / 90;
  gb = reduce_add(dZ, 0) / 90;
  W = W - lr * gW;
  b = b - lr * gb
}
logits = matmul(X, W) + matmul(ones([90, 1]), reshape(b, [1, 3]))
P = softmax(logits, 1)
pred = argmax(P, 1)
mean(eq(pred, tl))
"#;
    let acc = eval(src).data()[0];
    assert!(acc > 0.95, "accuracy {acc} below 0.95");
}
