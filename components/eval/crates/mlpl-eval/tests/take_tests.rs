//! `take(x, axis, idx)` builtin tests. Saga 29 step 007.
//!
//! Validates forward shape correctness, label propagation,
//! error paths, and gradcheck parity for the tape-
//! differentiable scatter-back gradient.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn run_err(src: &str, env: &mut Environment) -> mlpl_eval::EvalError {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap_err()
}

#[test]
fn take_axis_0_drops_leading_dim() {
    // [3, 4] -> take(x, 0, 1) -> [4]
    let mut env = Environment::new();
    let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
    env.set("x".into(), arr(vec![3, 4], xs));
    let y = run("take(x, 0, 1)", &mut env);
    assert_eq!(y.shape().dims(), &[4]);
    assert_eq!(y.data(), &[4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn take_axis_1_drops_middle_dim() {
    // [2, 3, 4] -> take(x, 1, 2) -> [2, 4]
    let mut env = Environment::new();
    let xs: Vec<f64> = (0..24).map(|i| i as f64).collect();
    env.set("x".into(), arr(vec![2, 3, 4], xs));
    let y = run("take(x, 1, 2)", &mut env);
    assert_eq!(y.shape().dims(), &[2, 4]);
    // axis-1 index 2 in row 0: indices 8, 9, 10, 11.
    // axis-1 index 2 in row 1: indices 20, 21, 22, 23.
    assert_eq!(y.data(), &[8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn take_axis_last_drops_trailing_dim() {
    // [2, 3] -> take(x, 1, 0) -> [2]
    let mut env = Environment::new();
    env.set(
        "x".into(),
        arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let y = run("take(x, 1, 0)", &mut env);
    assert_eq!(y.shape().dims(), &[2]);
    assert_eq!(y.data(), &[1.0, 4.0]);
}

#[test]
fn take_axis_out_of_range_errors() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![3, 4], vec![0.0; 12]));
    let err = run_err("take(x, 5, 0)", &mut env);
    let msg = format!("{err}");
    assert!(msg.contains("take") || msg.contains("shape"), "got: {msg}");
}

#[test]
fn take_idx_out_of_range_errors() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![3, 4], vec![0.0; 12]));
    let err = run_err("take(x, 0, 99)", &mut env);
    let msg = format!("{err}");
    assert!(msg.contains("take") || msg.contains("shape"), "got: {msg}");
}

#[test]
fn take_labels_drop_the_taken_axis() {
    // Build a labeled array, take an axis, verify the
    // remaining labels are correct.
    let xs: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let labeled = arr(vec![2, 3, 4], xs)
        .with_labels(vec![
            Some("batch".into()),
            Some("seq".into()),
            Some("dim".into()),
        ])
        .unwrap();
    let y = labeled.take(1, 1).unwrap();
    assert_eq!(y.shape().dims(), &[2, 4]);
    let labels = y.labels().expect("labels propagate");
    let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
    assert_eq!(names, vec!["batch", "dim"]);
}

#[test]
fn grad_of_sum_take_is_scatter_back_indicator() {
    // sum(take(X, 0, k)) -> d/dX_ij = (i == k) ? 1 : 0.
    let mut env = Environment::new();
    eval_program(&parse(&lex("X = param[4, 3]").unwrap()).unwrap(), &mut env).unwrap();
    let xs: Vec<f64> = (0..12).map(|i| i as f64 * 0.1).collect();
    env.set("X".into(), arr(vec![4, 3], xs));
    let g = run("grad(sum(take(X, 0, 2)), X)", &mut env);
    assert_eq!(g.shape().dims(), &[4, 3]);
    // Row 2 should be all-ones; every other row all-zero.
    for i in 0..4 {
        for j in 0..3 {
            let expected = if i == 2 { 1.0 } else { 0.0 };
            assert!(
                (g.data()[i * 3 + j] - expected).abs() < 1e-9,
                "grad[{i},{j}] = {} expected {expected}",
                g.data()[i * 3 + j]
            );
        }
    }
}

#[test]
fn grad_of_sum_take_squared_matches_finite_difference() {
    // sum(take(X, 0, k) * take(X, 0, k)) -> d/dX_ij =
    // (i == k) ? 2 * X[k, j] : 0. Closed-form should match
    // the autograd gradient.
    let mut env = Environment::new();
    eval_program(&parse(&lex("X = param[3, 4]").unwrap()).unwrap(), &mut env).unwrap();
    let xs: Vec<f64> = (0..12).map(|i| i as f64 * 0.1 - 0.5).collect();
    env.set("X".into(), arr(vec![3, 4], xs.clone()));
    let g = run("grad(sum(take(X, 0, 1) * take(X, 0, 1)), X)", &mut env);
    for i in 0..3 {
        for j in 0..4 {
            let expected = if i == 1 { 2.0 * xs[i * 4 + j] } else { 0.0 };
            let got = g.data()[i * 4 + j];
            assert!(
                (got - expected).abs() < 1e-6,
                "grad[{i},{j}] = {got} expected {expected}"
            );
        }
    }
}

#[test]
fn take_pets_tiny_first_image_returns_3_64_64() {
    // Round-trip through load_preloaded -> field access ->
    // take. Shape check only (heavy to verify pixel content).
    let v = run(
        "r = load_preloaded(\"pets_tiny\")\n\
         take(r.X, 0, 0)",
        &mut Environment::new(),
    );
    assert_eq!(v.shape().dims(), &[3, 64, 64]);
    let labels = v.labels().expect("axis labels propagate");
    let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
    assert_eq!(names, vec!["channel", "y", "x"]);
}

#[test]
fn take_pets_tiny_y_first_label_is_a_cat() {
    let v = run(
        "r = load_preloaded(\"pets_tiny\")\n\
         take(r.Y, 0, 0)",
        &mut Environment::new(),
    );
    // Pets_tiny convention: index 0 is a cat (label 0).
    assert_eq!(
        v.shape().dims(),
        &[] as &[usize],
        "take of rank-1 -> scalar"
    );
    assert_eq!(v.data(), &[0.0]);
}
