//! Saga 10, step 001: optimizer state scaffolding tests.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, OptimizerState, eval_program, optim_state, optim_state_mut};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn key(o: &str, p: &str, s: &str) -> (String, String, String) {
    (o.into(), p.into(), s.into())
}

#[test]
fn fresh_state_has_no_buffers_and_zero_steps() {
    let s = OptimizerState::default();
    assert!(s.buffers.is_empty());
    assert!(s.steps.is_empty());
}

#[test]
fn buffers_round_trip_through_environment() {
    let mut env = Environment::new();
    let v = arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    optim_state_mut(&mut env)
        .buffers
        .insert(key("momentum_sgd", "W", "v"), v.clone());
    let got = optim_state(&env)
        .buffers
        .get(&key("momentum_sgd", "W", "v"))
        .unwrap();
    assert_eq!(got.data(), v.data());
    // Different optimizer name = different slot, no collision.
    assert!(
        !optim_state(&env)
            .buffers
            .contains_key(&key("adam", "W", "v"))
    );
}

#[test]
fn step_counters_are_independent_per_optimizer() {
    let mut env = Environment::new();
    let steps = &mut optim_state_mut(&mut env).steps;
    *steps.entry("adam".into()).or_insert(0) += 1;
    *steps.entry("adam".into()).or_insert(0) += 1;
    *steps.entry("momentum_sgd".into()).or_insert(0) += 1;
    let s = optim_state(&env);
    assert_eq!(s.steps.get("adam").copied().unwrap_or(0), 2);
    assert_eq!(s.steps.get("momentum_sgd").copied().unwrap_or(0), 1);
}

#[test]
fn momentum_sgd_and_adam_are_recognized_but_stubbed() {
    // Step 001 only wires the dispatch -- both should hit a friendly
    // "not yet implemented" error rather than the generic "unknown
    // function" path.
    let mut env = Environment::new();
    eval_program(&parse(&lex("W = param[2]").unwrap()).unwrap(), &mut env).unwrap();

    for src in [
        "momentum_sgd(W, 0.1, 0.9)",
        "adam(W, 0.001, 0.9, 0.999, 0.00000001)",
    ] {
        let stmts = parse(&lex(src).unwrap()).unwrap();
        let err = eval_program(&stmts, &mut env).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not yet implemented"),
            "expected stub error for {src}, got {msg}"
        );
    }
}
