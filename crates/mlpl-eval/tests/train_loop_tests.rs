//! Saga 10 step 006: `train N { body }` training-loop sugar.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn train_records_step_index_in_last_losses() {
    // Body's final value is `step`, so last_losses == [0, 1, 2].
    let mut env = Environment::new();
    let stmts = parse(&lex("train 3 { step }").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();
    let losses = env
        .get("last_losses")
        .expect("last_losses bound after train");
    assert_eq!(losses.shape().dims(), &[3]);
    assert_eq!(losses.data(), &[0.0, 1.0, 2.0]);
}

#[test]
fn train_loss_decreases_with_adam() {
    // 50 Adam steps on sum(W*W) starting from W=[2.0]; loss must drop.
    let mut env = Environment::new();
    env.set_param("W".into(), arr(vec![1], vec![2.0]));
    // Inside adam() the loss expression uses tensor `sum`; the
    // captured per-step loss outside the optimizer uses runtime
    // `reduce_add` (since `sum` is autograd-only).
    let src = "train 50 { adam(sum(W*W), W, 0.1, 0.9, 0.999, 0.00000001); reduce_add(W*W) }";
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();
    let losses = env.get("last_losses").unwrap();
    assert_eq!(losses.shape().dims(), &[50]);
    let first = losses.data()[0];
    let last = losses.data()[49];
    assert!(
        last < first * 0.5,
        "expected loss to drop substantially, got first={first}, last={last}"
    );
    // And W should be near zero.
    let w = env.get("W").unwrap().data()[0];
    assert!(w.abs() < 0.5, "expected W ~= 0, got {w}");
}

#[test]
fn train_zero_steps_yields_empty_loss_history() {
    let mut env = Environment::new();
    let stmts = parse(&lex("train 0 { step }").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();
    let losses = env.get("last_losses").unwrap();
    assert_eq!(losses.shape().dims(), &[0]);
}
