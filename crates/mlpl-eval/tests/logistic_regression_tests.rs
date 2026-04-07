use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

fn eval_with_env(src: &str, env: &mut Environment) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

#[test]
fn logistic_regression_and_gate_converges() {
    // Train a logistic regression model on AND gate
    // After training, predictions should be close to [0, 0, 0, 1]
    let src = r#"
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]
w = zeros([2])
b = 0
lr = 1.0
n = 4
repeat 300 {
  z = matmul(X, reshape(w, [2, 1])) + b;
  pred = sigmoid(z);
  dz = pred - reshape(y, [4, 1]);
  dw = reshape(matmul(transpose(X), dz), [2]) / n;
  db = mean(dz);
  w = w - lr * dw;
  b = b - lr * db
}
pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)
pred
"#;
    let result = eval(src);
    let preds = result.data();
    assert_eq!(preds.len(), 4);
    // pred[0] (0,0) -> 0: should be < 0.3
    assert!(preds[0] < 0.3, "pred[0]={} should be < 0.3", preds[0]);
    // pred[1] (0,1) -> 0: should be < 0.3
    assert!(preds[1] < 0.3, "pred[1]={} should be < 0.3", preds[1]);
    // pred[2] (1,0) -> 0: should be < 0.3
    assert!(preds[2] < 0.3, "pred[2]={} should be < 0.3", preds[2]);
    // pred[3] (1,1) -> 1: should be > 0.7
    assert!(preds[3] > 0.7, "pred[3]={} should be > 0.7", preds[3]);
}

#[test]
fn logistic_regression_loss_decreases() {
    // Verify that loss decreases during training
    let mut env = Environment::new();

    // Setup
    eval_with_env("X = [[0,0],[0,1],[1,0],[1,1]]", &mut env);
    eval_with_env("y = [0, 0, 0, 1]", &mut env);
    eval_with_env("w = zeros([2])", &mut env);
    eval_with_env("b = 0", &mut env);
    eval_with_env("lr = 1.0", &mut env);
    eval_with_env("n = 4", &mut env);

    // Compute initial loss (MSE)
    let initial_loss = eval_with_env(
        "pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)\n\
         diff = pred - reshape(y, [4, 1])\n\
         mean(diff * diff)",
        &mut env,
    );

    // Train
    eval_with_env(
        "repeat 200 {\n\
           z = matmul(X, reshape(w, [2, 1])) + b;\n\
           pred = sigmoid(z);\n\
           dz = pred - reshape(y, [4, 1]);\n\
           dw = reshape(matmul(transpose(X), dz), [2]) / n;\n\
           db = mean(dz);\n\
           w = w - lr * dw;\n\
           b = b - lr * db\n\
         }",
        &mut env,
    );

    // Compute final loss (MSE)
    let final_loss = eval_with_env(
        "pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)\n\
         diff = pred - reshape(y, [4, 1])\n\
         mean(diff * diff)",
        &mut env,
    );

    let loss_i = initial_loss.data()[0];
    let loss_f = final_loss.data()[0];
    assert!(
        loss_f < loss_i,
        "Loss should decrease: initial={loss_i}, final={loss_f}"
    );
    assert!(loss_f < 0.05, "Final loss should be small: {loss_f}");
}

#[test]
fn logistic_regression_or_gate() {
    // OR gate: different problem, same structure
    let src = r#"
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 1, 1, 1]
w = zeros([2])
b = 0
lr = 1.0
n = 4
repeat 300 {
  z = matmul(X, reshape(w, [2, 1])) + b;
  pred = sigmoid(z);
  dz = pred - reshape(y, [4, 1]);
  dw = reshape(matmul(transpose(X), dz), [2]) / n;
  db = mean(dz);
  w = w - lr * dw;
  b = b - lr * db
}
pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)
pred
"#;
    let result = eval(src);
    let preds = result.data();
    // OR gate: (0,0)->0, others->1
    assert!(preds[0] < 0.3, "pred[0]={} should be < 0.3", preds[0]);
    assert!(preds[1] > 0.7, "pred[1]={} should be > 0.7", preds[1]);
    assert!(preds[2] > 0.7, "pred[2]={} should be > 0.7", preds[2]);
    assert!(preds[3] > 0.7, "pred[3]={} should be > 0.7", preds[3]);
}

#[test]
fn logistic_regression_accuracy_computation() {
    // Test that we can compute accuracy using eq and mean
    let src = r#"
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]
w = zeros([2])
b = 0
lr = 1.0
n = 4
repeat 300 {
  z = matmul(X, reshape(w, [2, 1])) + b;
  pred = sigmoid(z);
  dz = pred - reshape(y, [4, 1]);
  dw = reshape(matmul(transpose(X), dz), [2]) / n;
  db = mean(dz);
  w = w - lr * dw;
  b = b - lr * db
}
pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)
rounded = gt(pred, 0.5)
accuracy = mean(eq(reshape(rounded, [4]), y))
accuracy
"#;
    let result = eval(src);
    let acc = result.data()[0];
    assert!(acc >= 0.75, "Accuracy should be >= 75%: got {acc}");
}
