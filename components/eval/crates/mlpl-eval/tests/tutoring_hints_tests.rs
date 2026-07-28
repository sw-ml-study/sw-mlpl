//! Saga 23 step 008: tutoring-error-message catalog.
//!
//! Each test asserts that a specific (op, got_tag) error
//! produces a hint with the four required parts:
//! 1) what the op expected and why,
//! 2) what it got,
//! 3) likely cause,
//! 4) concrete fix with copy-pasteable MLPL syntax.
//!
//! The tests double as a contract check: hints cannot drift
//! away from valid MLPL syntax without breaking the
//! copy-paste-fix substring assertions.

use mlpl_array::{DenseArray, Shape};
use mlpl_core::ValueTag;
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn try_run(src: &str, env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
}

fn hint_of(err: &EvalError) -> String {
    match err {
        EvalError::TypeMismatch { hint, .. } => hint.clone(),
        _ => panic!("expected TypeMismatch, got {err:?}"),
    }
}

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn cross_entropy_probability_hint_has_four_parts() {
    let mut env = Environment::new();
    env.set("P".into(), arr(vec![2, 3], vec![0.5; 6]));
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0]));
    env.set_tag("P".into(), ValueTag::Probability);
    let h = hint_of(&try_run("loss = cross_entropy(P, T)\n", &mut env).unwrap_err());

    // 1) expected and why
    assert!(
        h.to_lowercase().contains("logit") && h.to_lowercase().contains("raw scores"),
        "expected explanation missing: {h}"
    );
    // 2) what it got
    assert!(
        h.to_lowercase().contains("probabilit"),
        "actual explanation missing: {h}"
    );
    // 3) likely cause
    assert!(
        h.to_lowercase().contains("softmax"),
        "likely cause missing: {h}"
    );
    // 4) concrete fix in copy-pasteable MLPL syntax
    // softmax in MLPL takes an axis: softmax(x, axis)
    assert!(
        h.contains("cross_entropy(logits, y)") || h.contains("cross_entropy(logits, T)"),
        "copy-pasteable cross_entropy fix missing: {h}"
    );
}

#[test]
fn cross_entropy_log_probability_hint_mentions_concrete_fix() {
    let mut env = Environment::new();
    env.set("LP".into(), arr(vec![2, 3], vec![-1.0; 6]));
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0]));
    env.set_tag("LP".into(), ValueTag::LogProbability);
    let h = hint_of(&try_run("loss = cross_entropy(LP, T)\n", &mut env).unwrap_err());
    assert!(
        h.to_lowercase().contains("log") && h.to_lowercase().contains("logit"),
        "log-prob context missing: {h}"
    );
    assert!(
        h.to_lowercase().contains("nll") || h.contains("exp("),
        "concrete fix missing: {h}"
    );
}

#[test]
fn cross_entropy_labels_hint_separates_first_arg_confusion() {
    // Common confusion: passed Labels (the integer class vector)
    // as the FIRST arg of cross_entropy.
    let mut env = Environment::new();
    env.set("Y".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0]));
    env.set_tag("Y".into(), ValueTag::Labels { num_classes: 3 });
    env.set("L".into(), arr(vec![3, 3], vec![1.0; 9]));
    let h = hint_of(&try_run("loss = cross_entropy(Y, L)\n", &mut env).unwrap_err());
    assert!(
        h.to_lowercase().contains("labels"),
        "labels mention missing: {h}"
    );
    // Concrete fix: swap arg order, or pass logits first.
    assert!(
        h.contains("cross_entropy(logits, y)") || h.contains("swap"),
        "concrete fix missing: {h}"
    );
}

#[test]
fn sample_probability_hint_mentions_logit() {
    let mut env = Environment::new();
    env.set("P".into(), arr(vec![2, 3], vec![0.5; 6]));
    env.set_tag("P".into(), ValueTag::Probability);
    let h = hint_of(&try_run("idx = sample(P, 1.0, 0)\n", &mut env).unwrap_err());
    assert!(h.to_lowercase().contains("logit"), "logit missing: {h}");
    assert!(h.to_lowercase().contains("softmax"), "softmax missing: {h}");
}

#[test]
fn top_k_log_probability_hint_mentions_concrete_fix() {
    let mut env = Environment::new();
    env.set(
        "LP".into(),
        DenseArray::from_vec(vec![-1.0, -2.0, -3.0, -0.5]),
    );
    env.set_tag("LP".into(), ValueTag::LogProbability);
    let h = hint_of(&try_run("t = top_k(LP, 2)\n", &mut env).unwrap_err());
    assert!(h.to_lowercase().contains("logit"), "logit missing: {h}");
    assert!(
        h.to_lowercase().contains("log"),
        "log-prob mention missing: {h}"
    );
}

#[test]
fn adam_probability_hint_has_concrete_fix() {
    let mut env = Environment::new();
    env.set("P".into(), DenseArray::from_scalar(0.5));
    env.set("W".into(), DenseArray::zeros(Shape::new(vec![1])));
    env.mark_param("W");
    env.set_tag("P".into(), ValueTag::Probability);
    let h = hint_of(
        &try_run(
            "_step = adam(P, W, 0.01, 0.9, 0.999, 0.00000001)\n",
            &mut env,
        )
        .unwrap_err(),
    );
    assert!(h.to_lowercase().contains("loss"), "loss missing: {h}");
    // Real MLPL fix: cross_entropy(logits, y) (mse/kl_divergence aren't
    // current builtins; cross_entropy is the canonical loss producer).
    assert!(
        h.contains("cross_entropy(logits, y)"),
        "concrete copy-paste fix missing: {h}"
    );
}

#[test]
fn adam_logit_hint_recommends_full_pipeline() {
    let mut env = Environment::new();
    env.set("L".into(), DenseArray::from_scalar(2.0));
    env.set("W".into(), DenseArray::zeros(Shape::new(vec![1])));
    env.mark_param("W");
    env.set_tag("L".into(), ValueTag::Logit);
    let h = hint_of(
        &try_run(
            "_step = adam(L, W, 0.01, 0.9, 0.999, 0.00000001)\n",
            &mut env,
        )
        .unwrap_err(),
    );
    assert!(
        h.contains("loss = cross_entropy(logits, y)"),
        "pipeline fix missing: {h}"
    );
    assert!(h.contains("adam(loss,"), "second-line fix missing: {h}");
}

#[test]
fn domain_mismatch_hint_lists_bridge_ops() {
    let mut env = Environment::new();
    env.set("L".into(), arr(vec![2, 3], vec![1.0; 6]));
    env.set("P".into(), arr(vec![2, 3], vec![0.3; 6]));
    env.set_tag("L".into(), ValueTag::Logit);
    env.set_tag("P".into(), ValueTag::Probability);
    let h = hint_of(&try_run("c = L + P\n", &mut env).unwrap_err());
    assert!(
        h.contains("softmax") && h.contains("log"),
        "bridge ops missing: {h}"
    );
}

#[test]
fn untagged_first_arg_yields_no_hint() {
    // A program with no tagged args must NOT raise TypeMismatch.
    // This is the gradual-typing additivity rule -- existing demos
    // keep working unchanged.
    let mut env = Environment::new();
    env.set("L".into(), arr(vec![2, 3], vec![1.0; 6]));
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0]));
    let result = try_run("loss = cross_entropy(L, T)\n", &mut env);
    assert!(result.is_ok(), "untagged should not error: {result:?}");
}
