//! Coverage for the "Naive Bayes (Gaussian)" demo (classical-ml saga
//! step 003): a generative classifier fit with NO training loop --
//! class masks, per-class means/variances via masked matmuls, and an
//! argmax over summed log-densities. Pins the fit statistics, the
//! accuracy-beats-baseline claim, and the train/test stability of the
//! closed-form fit on deterministic moons seeds.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

const NB_LINES: &[&str] = &[
    "M = moons(7, 100, 0.2)",
    "X = matmul(M, [[1, 0], [0, 1], [0, 0]]) ; Y = take(transpose(M), 0, 2)",
    "mask = one_hot(Y, 2) ; counts = reduce_add(mask, 0)",
    "mu = matmul(transpose(mask), X) / matmul(reshape(counts, [2, 1]), fill([1, 2], 1))",
    "s2 = matmul(transpose(mask), X * X) / matmul(reshape(counts, [2, 1]), fill([1, 2], 1)) - mu * mu + 0.001",
    "prior = counts / 100",
    "def u:loglik(P, c) { n = take(shape(P), 0, 0) ; MU = matmul(fill([n, 1], 1), reshape(take(mu, 0, c), [1, 2])) ; S2 = matmul(fill([n, 1], 1), reshape(take(s2, 0, c), [1, 2])) ; reduce_add(0 - 0.5 * log(2 * 3.141592653589793 * S2) - (P - MU) * (P - MU) / (2 * S2), 1) + log(take(prior, 0, c)) }",
    "T = moons(8, 60, 0.2) ; Xte = matmul(T, [[1, 0], [0, 1], [0, 0]]) ; Yte = take(transpose(T), 0, 2)",
    "pred = gt(u:loglik(Xte, 1), u:loglik(Xte, 0))",
];

fn run_lines(lines: &[&str]) -> Environment {
    let mut env = Environment::default();
    for line in lines {
        let toks = lex(line).expect("lex");
        let prog = parse(&toks).expect("parse");
        eval_program_value(&prog, &mut env).expect("eval");
    }
    env
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
fn gaussian_nb_beats_the_baseline_on_fresh_moons() {
    let mut env = run_lines(NB_LINES);
    let acc = eval_scalar(&mut env, "mean(eq(pred, Yte))");
    let base = eval_scalar(&mut env, "mean(eq(fill([60], 1), Yte))");
    // Moons are NOT Gaussian, so NB should be good-but-imperfect: well
    // above chance, below the ~1.0 an MLP reaches on the same data.
    assert!((0.75..0.97).contains(&acc), "NB accuracy on moons: {acc}");
    assert!(
        acc > base + 0.25,
        "NB ({acc}) vs one-class baseline ({base})"
    );
}

#[test]
fn closed_form_fit_is_stable_between_train_and_test() {
    let mut env = run_lines(NB_LINES);
    let train = eval_scalar(&mut env, "mean(eq(gt(u:loglik(X, 1), u:loglik(X, 0)), Y))");
    let test = eval_scalar(&mut env, "mean(eq(pred, Yte))");
    // No iterative training -> nothing to overfit; the two accuracies
    // should sit within a few points of each other.
    assert!(
        (train - test).abs() < 0.1,
        "closed-form fit should generalize: train {train} vs test {test}"
    );
}

#[test]
fn class_statistics_separate_the_two_moons() {
    let mut env = run_lines(NB_LINES);
    // Class 1's x-mean sits right of class 0's, and every variance is
    // positive (the 0.001 floor guards the log).
    let mu0x = eval_scalar(&mut env, "take(reshape(mu, [4]), 0, 0)");
    let mu1x = eval_scalar(&mut env, "take(reshape(mu, [4]), 0, 2)");
    assert!(
        mu1x > mu0x + 0.5,
        "x-means should separate: {mu0x} vs {mu1x}"
    );
    let all_pos = eval_scalar(&mut env, "reduce_mul(gt(reshape(s2, [4]), 0), 0)");
    assert!((all_pos - 1.0).abs() < 1e-9, "all variances positive");
}
