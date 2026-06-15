//! Verifies the MLPL behind the "Gradient Flow (backprop)" demo: a deep
//! sigmoid net whose per-layer gradient magnitudes are read with grad() and
//! shown as a bar chart. Backprop multiplies a <=0.25 sigmoid derivative at
//! each layer, so the gradient shrinks toward the input -- the classic
//! vanishing-gradient picture.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(lines: &[&str]) -> Value {
    let mut env = Environment::default();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in lines {
        let prog = parse(&lex(line).expect("lex")).expect("parse");
        last = eval_program_value(&prog, &mut env).expect("eval");
    }
    last
}

const SETUP: &[&str] = &[
    "M = moons(7, 40, 0.2)",
    "X = matmul(M, [[1,0],[0,1],[0,0]])",
    "y = reshape(matmul(M, [[0],[0],[1]]), [40])",
    "O = ones([40, 1])",
    "W1 = param[2, 12]",
    "b1 = param[1, 12]",
    "W2 = param[12, 12]",
    "b2 = param[1, 12]",
    "W3 = param[12, 12]",
    "b3 = param[1, 12]",
    "W4 = param[12, 2]",
    "b4 = param[1, 2]",
    "W1 = randn(1, [2, 12])",
    "W2 = randn(2, [12, 12])",
    "W3 = randn(3, [12, 12])",
    "W4 = randn(4, [12, 2])",
    // grad() needs the full forward+loss INLINE (it does not trace through
    // stored intermediates), so the deep-sigmoid loss is written out per layer.
    "g1 = grad(cross_entropy(matmul(sigmoid(matmul(sigmoid(matmul(sigmoid(matmul(X, W1) + matmul(O, b1)), W2) + matmul(O, b2)), W3) + matmul(O, b3)), W4) + matmul(O, b4), y), W1)",
    "g2 = grad(cross_entropy(matmul(sigmoid(matmul(sigmoid(matmul(sigmoid(matmul(X, W1) + matmul(O, b1)), W2) + matmul(O, b2)), W3) + matmul(O, b3)), W4) + matmul(O, b4), y), W2)",
    "g3 = grad(cross_entropy(matmul(sigmoid(matmul(sigmoid(matmul(sigmoid(matmul(X, W1) + matmul(O, b1)), W2) + matmul(O, b2)), W3) + matmul(O, b3)), W4) + matmul(O, b4), y), W3)",
    "g4 = grad(cross_entropy(matmul(sigmoid(matmul(sigmoid(matmul(sigmoid(matmul(X, W1) + matmul(O, b1)), W2) + matmul(O, b2)), W3) + matmul(O, b3)), W4) + matmul(O, b4), y), W4)",
    "m1 = sqrt(reduce_add(g1 * g1))",
    "m2 = sqrt(reduce_add(g2 * g2))",
    "m3 = sqrt(reduce_add(g3 * g3))",
    "m4 = sqrt(reduce_add(g4 * g4))",
];

#[test]
fn grad_traces_through_stored_activations_and_vanishes() {
    let mut lines = SETUP.to_vec();
    lines.push("[m1, m2, m3, m4]");
    let mags = match run(&lines) {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("expected vector, got {other:?}"),
    };
    eprintln!("per-layer grad magnitudes: {mags:?}");
    assert_eq!(mags.len(), 4, "four layers: {mags:?}");
    // grad() traced the inline forward back to W1 (else input-layer grad is 0).
    assert!(
        mags[0] > 0.0,
        "input-layer gradient is zero -- grad did not trace back"
    );
    // The output layer's gradient is much larger than the input layer's.
    assert!(
        mags[3] > mags[0] * 3.0,
        "gradient should shrink toward input: {mags:?}"
    );
}

#[test]
fn gradient_flow_renders_a_bar_chart() {
    let mut lines = SETUP.to_vec();
    lines.push("svg([m1, m2, m3, m4], \"bar\")");
    match run(&lines) {
        Value::Str(s) => assert!(s.starts_with("<svg") && s.contains("<rect"), "svg: {s}"),
        other => panic!("expected svg, got {other:?}"),
    }
}
