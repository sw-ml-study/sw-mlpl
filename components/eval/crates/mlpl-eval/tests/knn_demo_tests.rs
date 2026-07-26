//! Coverage for the "K-Nearest Neighbors" demo (classical-ml saga step
//! 002): the whole classifier is array algebra -- one matmul for every
//! test-train distance, argmax+one_hot rounds for the top-k vote --
//! with no training phase. Pins the demo program's data flow, the
//! |a|^2 + |b|^2 - 2ab identity against the pairwise_sqdist builtin,
//! and the accuracy-beats-baseline claim on deterministic seeds.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

const KNN_LINES: &[&str] = &[
    "B = blobs(7, 20, [[0, 0], [4, 4], [0 - 4, 4]])",
    "Xtr = matmul(B, [[1, 0], [0, 1], [0, 0]]) ; Ytr = take(transpose(B), 0, 2)",
    "T = blobs(8, 8, [[0, 0], [4, 4], [0 - 4, 4]]) ; Xte = matmul(T, [[1, 0], [0, 1], [0, 0]]) ; Yte = take(transpose(T), 0, 2)",
    "tr2 = reduce_add(Xtr * Xtr, 1) ; te2 = reduce_add(Xte * Xte, 1)",
    "D = matmul(reshape(te2, [24, 1]), fill([1, 60], 1)) + matmul(fill([24, 1], 1), reshape(tr2, [1, 60])) - 2 * matmul(Xte, transpose(Xtr))",
    "votes = fill([24, 3], 0) ; Dm = D ; for r in iota(5) { nn = argmax(0 - Dm, 1) ; sel = one_hot(nn, 60) ; votes = votes + matmul(sel, one_hot(Ytr, 3)) ; Dm = Dm + sel * 1000000 }",
    "pred = argmax(votes, 1)",
];

fn run_lines(lines: &[&str]) -> (Environment, Value) {
    let mut env = Environment::default();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in lines {
        let toks = lex(line).expect("lex");
        let prog = parse(&toks).expect("parse");
        last = eval_program_value(&prog, &mut env).expect("eval");
    }
    (env, last)
}

fn eval_scalar(env: &mut Environment, line: &str) -> f64 {
    let toks = lex(line).expect("lex");
    let prog = parse(&toks).expect("parse");
    match eval_program_value(&prog, env).expect("eval") {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn knn_beats_the_majority_baseline_on_blobs() {
    let (mut env, _) = run_lines(KNN_LINES);
    let acc = eval_scalar(&mut env, "mean(eq(pred, Yte))");
    let base = eval_scalar(&mut env, "mean(eq(fill([24], 0), Yte))");
    assert!(
        acc >= 0.9,
        "kNN should nail well-separated blobs, got {acc}"
    );
    assert!(
        acc > base + 0.3,
        "kNN ({acc}) should clearly beat always-one-class ({base})"
    );
}

#[test]
fn distance_identity_matches_pairwise_sqdist_builtin() {
    let (mut env, _) = run_lines(KNN_LINES);
    let err = eval_scalar(
        &mut env,
        "mean(abs(matmul(reshape(tr2, [60, 1]), fill([1, 60], 1)) + matmul(fill([60, 1], 1), reshape(tr2, [1, 60])) - 2 * matmul(Xtr, transpose(Xtr)) - pairwise_sqdist(Xtr)))",
    );
    assert!(
        err < 1e-9,
        "hand-built identity drifted from builtin: {err}"
    );
}

#[test]
fn vote_rounds_pick_five_distinct_neighbors() {
    // Each masking round must retire its pick: 5 rounds -> every row's
    // vote count sums to exactly 5.
    let (mut env, _) = run_lines(KNN_LINES);
    let total = eval_scalar(&mut env, "mean(reduce_add(votes, 1))");
    assert!(
        (total - 5.0).abs() < 1e-9,
        "each test point should cast exactly 5 votes, got {total}"
    );
}
