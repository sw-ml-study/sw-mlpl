//! Saga 23 step 009: smoke test for the "Typed ML Values"
//! tutorial lesson in apps/mlpl-web/src/lessons_advanced.rs.
//!
//! The lesson teaches typed values via runnable MLPL examples
//! that step through producer auto-tagging, the canonical
//! double-softmax bug, tag propagation, and :tags / :untag.
//! This test walks the same code paths through `eval_program`
//! and `inspect`, asserting the educational invariants the
//! lesson advertises. If a future refactor breaks one of
//! these, both the test and the lesson narrative must be
//! updated together.

use mlpl_array::DenseArray;
use mlpl_core::{LossKind, ValueTag};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, EvalError, eval_program, inspect};
use mlpl_parser::{lex, parse};

fn try_run(src: &str, env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
}

#[test]
fn lesson_step_1_softmax_produces_probability() {
    let mut env = Environment::new();
    try_run("L = randn(0, [2, 3])\n", &mut env).unwrap();
    try_run("probs = softmax(L, 1)\n", &mut env).unwrap();
    assert_eq!(env.get_tag("probs"), Some(&ValueTag::Probability));
    let desc = inspect(&mut env, ":describe probs").unwrap();
    assert!(desc.contains("Probability"), "describe: {desc}");
    assert!(
        desc.to_lowercase().contains("verified"),
        "row-sum invariant verified missing: {desc}"
    );
}

#[test]
fn lesson_step_2_cross_entropy_produces_loss() {
    let mut env = Environment::new();
    try_run("L = randn(0, [2, 3])\n", &mut env).unwrap();
    try_run("T = [0.0, 1.0]\n", &mut env).unwrap();
    try_run("loss = cross_entropy(L, T)\n", &mut env).unwrap();
    assert!(matches!(
        env.get_tag("loss"),
        Some(ValueTag::Loss {
            kind: LossKind::CrossEntropy
        })
    ));
}

#[test]
fn lesson_step_3_double_softmax_raises_tutoring_typemismatch() {
    let mut env = Environment::new();
    try_run("L = randn(0, [2, 3])\n", &mut env).unwrap();
    try_run("probs = softmax(L, 1)\n", &mut env).unwrap();
    try_run("T = [0.0, 1.0]\n", &mut env).unwrap();
    let err = try_run("loss = cross_entropy(probs, T)\n", &mut env).unwrap_err();
    match err {
        EvalError::TypeMismatch {
            op,
            expected,
            actual,
            hint,
        } => {
            assert_eq!(op, "cross_entropy");
            assert_eq!(expected, "Logit");
            assert_eq!(actual, "Probability");
            assert!(
                hint.contains("cross_entropy(logits, y)"),
                "hint must include canonical fix: {hint}"
            );
        }
        e => panic!("expected TypeMismatch, got {e:?}"),
    }
}

#[test]
fn lesson_step_4_grad_produces_gradient_with_wrt() {
    let mut env = Environment::new();
    try_run("W = param[3, 4]\n", &mut env).unwrap();
    try_run("X = randn(0, [2, 3])\n", &mut env).unwrap();
    try_run("Y = matmul(X, W)\n", &mut env).unwrap();
    try_run("g = grad(mean(Y), W)\n", &mut env).unwrap();
    assert_eq!(
        env.get_tag("g"),
        Some(&ValueTag::Gradient {
            wrt: "W".to_string()
        })
    );
}

#[test]
fn lesson_step_5_tag_propagates_through_addition() {
    let mut env = Environment::new();
    try_run("L1 = randn(3, [2, 3])\n", &mut env).unwrap();
    try_run("L2 = randn(4, [2, 3])\n", &mut env).unwrap();
    // Pretend the user has tagged L1 and L2 as Logit; the
    // lesson uses softmax-then-untag in the visible flow,
    // but the propagation rule fires regardless.
    env.set_tag("L1".into(), ValueTag::Logit);
    env.set_tag("L2".into(), ValueTag::Logit);
    try_run("sum_logits = L1 + L2\n", &mut env).unwrap();
    assert_eq!(env.get_tag("sum_logits"), Some(&ValueTag::Logit));
}

#[test]
fn lesson_step_6_logit_plus_probability_raises_domain_mismatch() {
    let mut env = Environment::new();
    try_run("L_tag = randn(5, [2, 3])\n", &mut env).unwrap();
    try_run("P_tag = softmax(L_tag, 1)\n", &mut env).unwrap();
    env.set_tag("L_tag".into(), ValueTag::Logit);
    let err = try_run("mix = L_tag + P_tag\n", &mut env).unwrap_err();
    assert!(matches!(err, EvalError::TypeMismatch { .. }), "got {err:?}");
}

#[test]
fn lesson_step_7_apply_softmax_layer_tail_yields_probability() {
    let mut env = Environment::new();
    try_run("mdl = chain(linear(3, 4, 0), softmax_layer())\n", &mut env).unwrap();
    try_run("out = apply(mdl, randn(0, [2, 3]))\n", &mut env).unwrap();
    assert_eq!(env.get_tag("out"), Some(&ValueTag::Probability));
}

#[test]
fn lesson_step_8_untag_clears_tag() {
    let mut env = Environment::new();
    try_run("x = randn(0, [2, 3])\n", &mut env).unwrap();
    env.set_tag("x".into(), ValueTag::Logit);
    let out = inspect(&mut env, ":untag x").unwrap();
    assert!(out.to_lowercase().contains("untagged"), "out: {out}");
    assert!(env.get_tag("x").is_none(), "tag should be cleared");
}

#[test]
fn lesson_step_9_tags_lists_tagged_bindings() {
    let mut env = Environment::new();
    try_run("L = randn(0, [2, 3])\n", &mut env).unwrap();
    try_run("probs = softmax(L, 1)\n", &mut env).unwrap();
    let out = inspect(&mut env, ":tags").unwrap();
    assert!(out.contains("probs"), "probs should appear in :tags: {out}");
    assert!(
        out.contains("Probability"),
        "Probability tag should appear: {out}"
    );
}
