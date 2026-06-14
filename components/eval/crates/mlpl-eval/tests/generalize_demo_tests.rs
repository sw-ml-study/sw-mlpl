//! Coverage for the "Watch a Model Generalize (no overfitting)" demo: an
//! ample training set + a right-sized single-hidden-layer net, so train and
//! validation loss descend together and the model generalizes. The miniature
//! proxy (fast, dev) checks the program is valid MLPL and returns a sane
//! accuracy; the `#[ignore]` full test (run in --release) asserts the real
//! demo actually reaches good validation accuracy.

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

#[test]
fn generalize_structure_runs_in_miniature() {
    // 40 points, one chunk of 10 steps: just confirm the apply/adam/concat/
    // accuracy data-flow is valid and returns an accuracy in [0, 1].
    let acc = scalar(run_lines(&[
        "Mtr = moons(7, 40, 0.2)",
        "Xtr = matmul(Mtr, [[1,0],[0,1],[0,0]])",
        "ytr = reshape(matmul(Mtr, [[0],[0],[1]]), [40])",
        "Mva = moons(23, 40, 0.2)",
        "Xva = matmul(Mva, [[1,0],[0,1],[0,0]])",
        "yva = reshape(matmul(Mva, [[0],[0],[1]]), [40])",
        "mdl = chain(linear(2, 16, 1), tanh_layer(), linear(16, 2, 2))",
        "train 10 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "preds = argmax(softmax(apply(mdl, Xva), 1), 1)",
        "mean(eq(preds, yva))",
    ]));
    assert!((0.0..=1.0).contains(&acc), "accuracy out of range: {acc}");
}

#[test]
#[ignore = "heavy: 100 Adam steps on 200 points; run with --release --ignored"]
fn watch_model_generalize_reaches_good_accuracy() {
    // The full demo's training, ending in validation accuracy. A right-sized
    // net on ample data should generalize well to the unseen validation set.
    let acc = scalar(run_lines(&[
        "Mtr = moons(7, 200, 0.20)",
        "Xtr = matmul(Mtr, [[1,0],[0,1],[0,0]])",
        "ytr = reshape(matmul(Mtr, [[0],[0],[1]]), [200])",
        "Mva = moons(23, 200, 0.20)",
        "Xva = matmul(Mva, [[1,0],[0,1],[0,0]])",
        "yva = reshape(matmul(Mva, [[0],[0],[1]]), [200])",
        "mdl = chain(linear(2, 16, 1), tanh_layer(), linear(16, 2, 2))",
        "train 20 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "train 20 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "train 20 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "train 20 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "train 20 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.06, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }",
        "preds = argmax(softmax(apply(mdl, Xva), 1), 1)",
        "mean(eq(preds, yva))",
    ]));
    assert!(
        acc >= 0.85,
        "expected good generalization, got val accuracy {acc}"
    );
}
