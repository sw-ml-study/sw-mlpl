//! Saga 23 step 004: predicate-checked consumers.
//!
//! `cross_entropy` / `sample` / `top_k` reject `Probability` and
//! `LogProbability` inputs in arg 0; `adam` / `momentum_sgd`
//! reject non-`Loss` inputs in arg 0. Untyped values (no tag in
//! the side table) pass every predicate (gradual-typing
//! additivity).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn try_run(src: &str, env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    try_run(src, env).unwrap()
}

fn type_mismatch_msg(err: &EvalError) -> Option<String> {
    match err {
        EvalError::TypeMismatch {
            op,
            expected,
            actual,
            hint,
        } => Some(format!(
            "op={op} expected={expected} actual={actual} hint={hint}"
        )),
        _ => None,
    }
}

#[test]
fn cross_entropy_accepts_logit_named_binding() {
    let mut env = Environment::new();
    env.set(
        "L".to_string(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 0.5, 0.4, 0.1]).unwrap(),
    );
    env.set("T".to_string(), DenseArray::from_vec(vec![0.0, 2.0]));
    // Tag L as Logit; cross_entropy should accept.
    env.set_tag("L".to_string(), mlpl_core::ValueTag::Logit);
    run("loss = cross_entropy(L, T)\n", &mut env);
}

#[test]
fn cross_entropy_rejects_probability_named_binding_with_hint() {
    let mut env = Environment::new();
    env.set(
        "P".to_string(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.1, 0.7, 0.2, 0.4, 0.4, 0.2]).unwrap(),
    );
    env.set("T".to_string(), DenseArray::from_vec(vec![1.0, 0.0]));
    env.set_tag("P".to_string(), mlpl_core::ValueTag::Probability);
    let err = try_run("loss = cross_entropy(P, T)\n", &mut env).unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("cross_entropy"), "msg: {msg}");
    assert!(msg.contains("Logit"), "msg: {msg}");
    assert!(msg.contains("Probability"), "msg: {msg}");
    // Hint should be tutoring -- mention raw scores, softmax, or nll.
    assert!(
        msg.to_lowercase().contains("softmax") || msg.to_lowercase().contains("raw scores"),
        "hint should mention softmax or raw scores, got: {msg}"
    );
}

#[test]
fn cross_entropy_rejects_inline_softmax_with_hint() {
    // The canonical double-softmax bug: cross_entropy(softmax(logits), Y).
    let mut env = Environment::new();
    env.set(
        "L".to_string(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap(),
    );
    env.set("T".to_string(), DenseArray::from_vec(vec![0.0, 1.0]));
    let err = try_run("loss = cross_entropy(softmax(L, 1), T)\n", &mut env).unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("cross_entropy"), "msg: {msg}");
    assert!(msg.contains("Probability"), "msg: {msg}");
}

#[test]
fn cross_entropy_passes_untagged_logit_shape_array() {
    // The user passes a raw [N, V] array with no tag set; predicate
    // should NOT fire (gradual-typing additivity).
    let mut env = Environment::new();
    env.set(
        "L".to_string(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap(),
    );
    env.set("T".to_string(), DenseArray::from_vec(vec![0.0, 1.0]));
    run("loss = cross_entropy(L, T)\n", &mut env);
}

#[test]
fn sample_rejects_probability_input_with_hint() {
    let mut env = Environment::new();
    env.set(
        "P".to_string(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.1, 0.7, 0.2, 0.4, 0.4, 0.2]).unwrap(),
    );
    env.set_tag("P".to_string(), mlpl_core::ValueTag::Probability);
    let err = try_run("idx = sample(P, 1.0, 0)\n", &mut env).unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("sample"), "msg: {msg}");
    assert!(msg.contains("Logit"), "msg: {msg}");
}

#[test]
fn top_k_rejects_probability_input_with_hint() {
    let mut env = Environment::new();
    env.set(
        "P".to_string(),
        DenseArray::from_vec(vec![0.1, 0.4, 0.3, 0.2]),
    );
    env.set_tag("P".to_string(), mlpl_core::ValueTag::Probability);
    let err = try_run("t = top_k(P, 2)\n", &mut env).unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("top_k"), "msg: {msg}");
}

#[test]
fn adam_rejects_non_loss_first_arg_with_hint() {
    let mut env = Environment::new();
    let src = "W = param[2, 3]\nX = randn(0, [4, 2])\nY = matmul(X, W)\nloss = mean(Y)\n";
    run(src, &mut env);
    // Manually clear the loss tag so it doesn't match Loss; pretend
    // the user typoed and passed a Probability instead.
    env.clear_tag("loss");
    env.set_tag("loss".to_string(), mlpl_core::ValueTag::Probability);
    let err = try_run(
        "_step = adam(loss, W, 0.01, 0.9, 0.999, 0.00000001)\n",
        &mut env,
    )
    .unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("adam"), "msg: {msg}");
    assert!(msg.contains("Loss"), "msg: {msg}");
}

#[test]
fn adam_accepts_loss_first_arg() {
    // mean of an untagged array won't auto-tag Loss, but the producer
    // chain `loss = cross_entropy(...)` does. The full grad/adam
    // pipeline through a Loss-tagged scalar should succeed.
    let mut env = Environment::new();
    let src = "W = param[2, 3]\nX = randn(0, [4, 2])\nY = matmul(X, W)\nP = softmax(Y, 1)\n";
    run(src, &mut env);
    // Build a synthetic Loss-tagged scalar.
    env.set("loss".to_string(), DenseArray::from_scalar(1.234));
    env.set_tag(
        "loss".to_string(),
        mlpl_core::ValueTag::Loss {
            kind: mlpl_core::LossKind::CrossEntropy,
        },
    );
    // adam either succeeds or fails for some other reason (e.g.
    // missing grad), but it MUST NOT raise TypeMismatch -- the
    // Loss tag passes the predicate.
    match try_run(
        "_step = adam(loss, W, 0.01, 0.9, 0.999, 0.00000001)\n",
        &mut env,
    ) {
        Ok(_) => {}
        Err(err) => assert!(
            type_mismatch_msg(&err).is_none(),
            "Loss-tagged loss should pass the type predicate; got TypeMismatch: {err:?}"
        ),
    }
}

#[test]
fn momentum_sgd_rejects_logit_first_arg_with_hint() {
    let mut env = Environment::new();
    env.set("L".to_string(), DenseArray::from_scalar(2.5));
    env.set_tag("L".to_string(), mlpl_core::ValueTag::Logit);
    env.set("W".to_string(), DenseArray::zeros(Shape::new(vec![1])));
    env.mark_param("W");
    let err = try_run("_step = momentum_sgd(L, W, 0.01, 0.9)\n", &mut env).unwrap_err();
    let msg = type_mismatch_msg(&err).expect("expected TypeMismatch");
    assert!(msg.contains("momentum_sgd"), "msg: {msg}");
    assert!(msg.contains("Loss"), "msg: {msg}");
}
