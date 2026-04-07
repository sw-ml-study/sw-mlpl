//! Integration test: a tiny 2 -> 8 -> 2 MLP outperforms a linear
//! softmax classifier on an XOR-style non-linearly separable dataset.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval_with(env: &mut Environment, src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

#[test]
fn tiny_mlp_beats_linear_on_xor_blobs() {
    let mut env = Environment::new();
    // XOR-style 4-blob dataset: classes by diagonal.
    eval_with(
        &mut env,
        r#"
D  = blobs(3, 20, [[-2,-2],[2,2],[-2,2],[2,-2]])
X  = matmul(D, [[1,0],[0,1],[0,0]])
raw = reshape(matmul(D, [[0],[0],[1]]), [80])
y  = gt(raw, 1.5)
Y  = one_hot(y, 2)
"#,
    );

    // ---- linear softmax baseline ----
    eval_with(
        &mut env,
        r#"
W = zeros([2, 2])
b = zeros([2])
lr = 0.3
repeat 300 {
  logits = matmul(X, W) + matmul(ones([80, 1]), reshape(b, [1, 2]));
  P = softmax(logits, 1);
  dZ = P - Y;
  gW = matmul(transpose(X), dZ) / 80;
  gb = reduce_add(dZ, 0) / 80;
  W = W - lr * gW;
  b = b - lr * gb
}
logits = matmul(X, W) + matmul(ones([80, 1]), reshape(b, [1, 2]))
P = softmax(logits, 1)
pred = argmax(P, 1)
lin_acc = mean(eq(pred, y))
"#,
    );

    // ---- tiny MLP ----
    eval_with(
        &mut env,
        r#"
W1 = randn(5, [2, 8]) * 0.5
b1 = zeros([8])
W2 = randn(6, [8, 2]) * 0.5
b2 = zeros([2])
lr = 0.2
repeat 600 {
  Z1 = matmul(X, W1) + matmul(ones([80, 1]), reshape(b1, [1, 8]));
  H  = tanh_fn(Z1);
  Z2 = matmul(H, W2) + matmul(ones([80, 1]), reshape(b2, [1, 2]));
  P  = softmax(Z2, 1);
  dZ2 = P - Y;
  gW2 = matmul(transpose(H), dZ2) / 80;
  gb2 = reduce_add(dZ2, 0) / 80;
  dH  = matmul(dZ2, transpose(W2));
  dZ1 = dH * (1 - H * H);
  gW1 = matmul(transpose(X), dZ1) / 80;
  gb1 = reduce_add(dZ1, 0) / 80;
  W1 = W1 - lr * gW1;
  b1 = b1 - lr * gb1;
  W2 = W2 - lr * gW2;
  b2 = b2 - lr * gb2
}
Z1 = matmul(X, W1) + matmul(ones([80, 1]), reshape(b1, [1, 8]))
H  = tanh_fn(Z1)
Z2 = matmul(H, W2) + matmul(ones([80, 1]), reshape(b2, [1, 2]))
P  = softmax(Z2, 1)
pred = argmax(P, 1)
mlp_acc = mean(eq(pred, y))
"#,
    );

    let lin_acc = eval_with(&mut env, "lin_acc").data()[0];
    let mlp_acc = eval_with(&mut env, "mlp_acc").data()[0];
    assert!(
        lin_acc < 0.75,
        "linear should struggle on XOR, got {lin_acc}"
    );
    assert!(
        mlp_acc >= lin_acc,
        "mlp {mlp_acc} should beat linear {lin_acc}"
    );
    assert!(mlp_acc > 0.90, "mlp accuracy {mlp_acc} should exceed 0.90");
}
