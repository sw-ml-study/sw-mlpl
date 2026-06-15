//! Verifies the MLPL behind two new Training & Learning demos:
//! - "Taming Overfitting: Weight Decay" -- a hand-rolled over-capacity net on
//!   tiny noisy data with an L2 penalty added to the loss.
//! - "Learning Rate: too big, too small, just right" -- gradient descent on
//!   one 2-weight loss surface at three rates.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run_lines(lines: &[&str]) -> Value {
    let mut env = Environment::default();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in lines {
        let toks = lex(line).expect("lex");
        let prog = parse(&toks).expect("parse");
        last = eval_program_value(&prog, &mut env).expect("eval");
    }
    last
}

fn scalar(v: Value) -> f64 {
    match v {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

const FWD30: &str = "matmul(tanh_fn(matmul(Xtr, W1) + matmul(O30, b1)), W2) + matmul(O30, b2)";
const FWD200: &str = "matmul(tanh_fn(matmul(Xva, W1) + matmul(O200, b1)), W2) + matmul(O200, b2)";

#[test]
fn weight_decay_miniature_runs() {
    // Tiny version (2 chunks x 5 steps) of the demo: hand-rolled net + L2
    // penalty. Just confirm the data-flow is valid and returns an accuracy.
    let train = format!(
        "train 5 {{ adam(cross_entropy({FWD30}, ytr) + lam * (sum(W1 * W1) + sum(W2 * W2)), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); cross_entropy({FWD30}, ytr) }}"
    );
    let acc = scalar(run_lines(&[
        "Mtr = moons(7, 30, 0.30)",
        "Xtr = matmul(Mtr, [[1,0],[0,1],[0,0]])",
        "ytr = reshape(matmul(Mtr, [[0],[0],[1]]), [30])",
        "Mva = moons(101, 200, 0.30)",
        "Xva = matmul(Mva, [[1,0],[0,1],[0,0]])",
        "yva = reshape(matmul(Mva, [[0],[0],[1]]), [200])",
        "O30 = ones([30, 1])",
        "O200 = ones([200, 1])",
        "W1 = param[2, 32]",
        "b1 = param[1, 32]",
        "W2 = param[32, 2]",
        "b2 = param[1, 2]",
        "W1 = randn(11, [2, 32]) * 0.5",
        "W2 = randn(12, [32, 2]) * 0.5",
        "lam = 0.02",
        &train,
        &train,
        &format!("preds = argmax(softmax({FWD200}, 1), 1)"),
        "mean(eq(preds, yva))",
    ]));
    assert!((0.0..=1.0).contains(&acc), "accuracy out of range: {acc}");
}

#[test]
#[ignore = "heavy: 150 Adam steps; run with --release --ignored"]
fn weight_decay_full_generalizes() {
    // The full demo's training: an over-capacity net on 30 noisy points with
    // an L2 penalty should still generalize (the penalty stops it memorizing).
    let train = format!(
        "train 25 {{ adam(cross_entropy({FWD30}, ytr) + lam * (sum(W1 * W1) + sum(W2 * W2)), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); cross_entropy({FWD30}, ytr) }}"
    );
    let acc = scalar(run_lines(&[
        "Mtr = moons(7, 30, 0.30)",
        "Xtr = matmul(Mtr, [[1,0],[0,1],[0,0]])",
        "ytr = reshape(matmul(Mtr, [[0],[0],[1]]), [30])",
        "Mva = moons(101, 200, 0.30)",
        "Xva = matmul(Mva, [[1,0],[0,1],[0,0]])",
        "yva = reshape(matmul(Mva, [[0],[0],[1]]), [200])",
        "O30 = ones([30, 1])",
        "O200 = ones([200, 1])",
        "W1 = param[2, 32]",
        "b1 = param[1, 32]",
        "W2 = param[32, 2]",
        "b2 = param[1, 2]",
        "W1 = randn(11, [2, 32]) * 0.5",
        "W2 = randn(12, [32, 2]) * 0.5",
        "lam = 0.03",
        &train,
        &train,
        &train,
        &train,
        &train,
        &train,
        &format!("preds = argmax(softmax({FWD200}, 1), 1)"),
        "mean(eq(preds, yva))",
    ]));
    assert!(
        acc >= 0.78,
        "weight decay should still generalize, got {acc}"
    );
}

#[test]
fn learning_rate_too_big_renders_without_crashing() {
    // The lr=0.7 trajectory diverges (w blows up); loss_landscape must clamp
    // the runaway path coords and still produce a valid SVG, not NaN garbage.
    let v = run_lines(&[
        "x = linspace(-2, 2, 11)",
        "y = 2 * x + 1",
        "G = grid([-1, 5, -3, 3], 8)",
        "wcol = reshape(take(transpose(G), 0, 0), [64, 1])",
        "bcol = reshape(take(transpose(G), 0, 1), [64, 1])",
        "preds = matmul(wcol, reshape(x, [1, 11])) + matmul(bcol, ones([1, 11]))",
        "resid = preds - matmul(ones([64, 1]), reshape(y, [1, 11]))",
        "surf = reduce_add(resid * resid, 1) / 11",
        "w = 4.5",
        "b = -2.5",
        "path = [[w, b]]",
        "repeat 40 { e = (w * x + b) - y; dw = mean(e * x) * 2; db = mean(e) * 2; w = w - 0.7 * dw; b = b - 0.7 * db; path = concat(path, [[w, b]], 0) }",
        "pathn = (path - matmul(ones([41, 1]), [[-1, -3]])) / matmul(ones([41, 1]), [[6, 6]])",
        "loss_landscape(surf, [8, 8], pathn)",
    ]);
    match v {
        Value::Str(s) => assert!(s.starts_with("<svg") && !s.contains("NaN"), "svg: {s}"),
        other => panic!("expected svg, got {other:?}"),
    }
}

fn lr_final(lr: &str) -> (f64, f64) {
    let step = format!(
        "repeat 40 {{ e = (w * x + b) - y; dw = mean(e * x) * 2; db = mean(e) * 2; w = w - {lr} * dw; b = b - {lr} * db }}"
    );
    let mut env = Environment::default();
    for line in [
        "x = linspace(-2, 2, 11)",
        "y = 2 * x + 1",
        "w = 4.5",
        "b = -2.5",
        &step,
    ] {
        let prog = parse(&lex(line).unwrap()).unwrap();
        eval_program_value(&prog, &mut env).unwrap();
    }
    let w = scalar(eval_program_value(&parse(&lex("w").unwrap()).unwrap(), &mut env).unwrap());
    let b = scalar(eval_program_value(&parse(&lex("b").unwrap()).unwrap(), &mut env).unwrap());
    (w, b)
}

#[test]
fn learning_rate_three_regimes_behave() {
    // Truth is w=2, b=1. Too-small barely moves from (4.5,-2.5); just-right
    // converges near (2,1); too-big overshoots (does not settle near (2,1)).
    let small = lr_final("0.003");
    let good = lr_final("0.15");
    let big = lr_final("0.7");
    eprintln!("small={small:?} good={good:?} big={big:?}");
    assert!(small.0 > 3.0, "too-small should barely move w: {small:?}");
    assert!(
        (good.0 - 2.0).abs() < 0.3 && (good.1 - 1.0).abs() < 0.3,
        "good should converge: {good:?}"
    );
    assert!(
        (big.0 - 2.0).abs() > 2.0,
        "too-big should diverge away from w=2: {big:?}"
    );
}
