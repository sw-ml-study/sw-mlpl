//! Saga 15 step 001: `freeze(m)` / `unfreeze(m)` +
//! `env.frozen_params`.
//!
//! Groundwork for LoRA. `freeze(m)` marks every one of `m`'s
//! parameters in `env.frozen_params`; `adam` and
//! `momentum_sgd` skip any name in that set at the
//! optimizer-update stage. Gradients still flow through
//! frozen parameters -- freezing is strictly an optimizer-
//! side filter, not a gradient mask.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

fn snapshot_params(env: &Environment, model_name: &str) -> Vec<(String, Vec<f64>)> {
    model_params(env, model_name)
        .unwrap()
        .into_iter()
        .map(|n| {
            let v = env.get(&n).unwrap().data().to_vec();
            (n, v)
        })
        .collect()
}

#[test]
fn freeze_marks_every_param_in_frozen_set() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    run(&mut env, "freeze(m)");
    let names = model_params(&env, "m").unwrap();
    for n in &names {
        assert!(
            env.is_frozen(n),
            "param '{n}' should be frozen after freeze(m)"
        );
        assert!(
            env.is_param(n),
            "param '{n}' must still be a trainable name (frozen is an overlay, not a replacement)"
        );
    }
}

#[test]
fn adam_leaves_frozen_model_bit_identical() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    env.set("Y".into(), arr(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5]));

    let before = snapshot_params(&env, "m");
    run(&mut env, "freeze(m)");
    run(
        &mut env,
        "train 3 { adam(mean((apply(m, X) - Y) * (apply(m, X) - Y)), m, 0.1, 0.9, 0.999, 0.00000001); loss_metric = mean((apply(m, X) - Y) * (apply(m, X) - Y)) }",
    );
    let after = snapshot_params(&env, "m");

    assert_eq!(
        before, after,
        "frozen model params must be bit-identical after 3 adam steps"
    );
}

#[test]
fn unfreeze_restores_adam_updates() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    env.set("Y".into(), arr(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5]));

    run(&mut env, "freeze(m)");
    run(&mut env, "unfreeze(m)");

    let before = snapshot_params(&env, "m");
    run(
        &mut env,
        "train 3 { adam(mean((apply(m, X) - Y) * (apply(m, X) - Y)), m, 0.1, 0.9, 0.999, 0.00000001); loss_metric = mean((apply(m, X) - Y) * (apply(m, X) - Y)) }",
    );
    let after = snapshot_params(&env, "m");

    assert_ne!(before, after, "unfrozen model must move under adam");
}

#[test]
fn partial_freeze_isolates_training_to_unfrozen_model() {
    let mut env = Environment::new();
    run(&mut env, "m1 = linear(2, 2, 1)");
    run(&mut env, "m2 = linear(2, 2, 2)");
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    env.set("Y".into(), arr(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5]));

    let before_m1 = snapshot_params(&env, "m1");
    let before_m2 = snapshot_params(&env, "m2");

    run(&mut env, "freeze(m1)");
    // Train only m2.
    run(
        &mut env,
        "train 3 { adam(mean((apply(m2, X) - Y) * (apply(m2, X) - Y)), m2, 0.1, 0.9, 0.999, 0.00000001); loss_metric = mean((apply(m2, X) - Y) * (apply(m2, X) - Y)) }",
    );

    let after_m1 = snapshot_params(&env, "m1");
    let after_m2 = snapshot_params(&env, "m2");

    assert_eq!(before_m1, after_m1, "frozen m1 must stay identical");
    assert_ne!(before_m2, after_m2, "unfrozen m2 must move");
}

#[test]
fn momentum_sgd_also_respects_frozen() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    env.set("Y".into(), arr(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5]));

    let before = snapshot_params(&env, "m");
    run(&mut env, "freeze(m)");
    run(
        &mut env,
        "train 3 { momentum_sgd(mean((apply(m, X) - Y) * (apply(m, X) - Y)), m, 0.1, 0.9); loss_metric = mean((apply(m, X) - Y) * (apply(m, X) - Y)) }",
    );
    let after = snapshot_params(&env, "m");
    assert_eq!(before, after, "momentum_sgd must honor freeze too");
}

#[test]
fn freeze_is_idempotent() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    run(&mut env, "freeze(m)");
    run(&mut env, "freeze(m)");
    // No error; every name is still frozen.
    for n in model_params(&env, "m").unwrap() {
        assert!(env.is_frozen(&n));
    }
}

#[test]
fn freeze_rejects_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    let stmts = parse(&lex("freeze(m, m)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity mismatch");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("freeze") || msg.contains("arity"),
        "error should mention freeze / arity, got: {msg}"
    );
}

#[test]
fn freeze_rejects_non_model_argument() {
    let mut env = Environment::new();
    run(&mut env, "x = 1");
    let stmts = parse(&lex("freeze(x)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("x is not a model");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("freeze") || msg.contains("model"),
        "error should mention freeze / model, got: {msg}"
    );
}

#[test]
fn unfreeze_rejects_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(2, 2, 1)");
    let stmts = parse(&lex("unfreeze()").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity mismatch");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("unfreeze") || msg.contains("arity"),
        "error should mention unfreeze / arity, got: {msg}"
    );
}
