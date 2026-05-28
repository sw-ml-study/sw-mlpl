//! Saga 20 step 003: `argtop_k(values, k)`.
//!
//! Index-returning companion to the existing `top_k(logits, k)`
//! (which masks logits in place). `argtop_k` returns the k indices
//! of the largest entries in a rank-1 float vector, sorted by
//! descending value and with ties broken by lower index first.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run_prog(env: &mut Environment, src: &str) -> DenseArray {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap()
}

#[test]
fn argtop_k_returns_indices_of_largest_entries_sorted_by_value() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![4], vec![0.1, 0.5, 0.2, 0.9]));
    let out = run_prog(&mut env, "argtop_k(v, 2)");
    assert_eq!(out.shape().dims(), &[2]);
    // Sorted by descending value: 0.9 at idx 3, then 0.5 at idx 1.
    assert_eq!(out.data(), &[3.0, 1.0]);
}

#[test]
fn argtop_k_breaks_ties_by_lower_index_first() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![3], vec![1.0, 1.0, 0.0]));
    let out = run_prog(&mut env, "argtop_k(v, 2)");
    assert_eq!(out.data(), &[0.0, 1.0], "lower index wins tie");
}

#[test]
fn argtop_k_with_k_equal_to_length_returns_full_sort() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![4], vec![0.9, 0.1, 0.5, 0.2]));
    let out = run_prog(&mut env, "argtop_k(v, 4)");
    assert_eq!(out.data(), &[0.0, 2.0, 3.0, 1.0]);
}

#[test]
fn argtop_k_single_element_k_one() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![1], vec![5.0]));
    let out = run_prog(&mut env, "argtop_k(v, 1)");
    assert_eq!(out.data(), &[0.0]);
}

#[test]
fn argtop_k_rejects_k_greater_than_length() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![3], vec![0.1, 0.2, 0.3]));
    let stmts = parse(&lex("argtop_k(v, 5)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("k=5 > len=3 should error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("argtop_k") && (msg.contains("5") || msg.contains('3')),
        "error should name argtop_k and indicate the mismatch, got: {msg}"
    );
}

#[test]
fn argtop_k_rejects_non_rank_1_input() {
    let mut env = Environment::new();
    env.set("m".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let stmts = parse(&lex("argtop_k(m, 2)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("non-rank-1 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("argtop_k") && (msg.contains("rank") || msg.contains("vector")),
        "error should mention rank/vector, got: {msg}"
    );
}

#[test]
fn argtop_k_rejects_negative_k() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![3], vec![0.1, 0.2, 0.3]));
    let stmts = parse(&lex("argtop_k(v, -1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("negative k should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("argtop_k"),
        "error should mention argtop_k, got: {msg}"
    );
}

#[test]
fn argtop_k_rejects_wrong_arity() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![3], vec![0.1, 0.2, 0.3]));
    let stmts = parse(&lex("argtop_k(v)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("1 arg should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("argtop_k")
            && (msg.contains("arity") || msg.contains("expected") || msg.contains("args")),
        "error should reference argtop_k arity, got: {msg}"
    );
}

#[test]
fn argtop_k_k_zero_returns_empty_vector() {
    let mut env = Environment::new();
    env.set("v".into(), arr(vec![3], vec![0.1, 0.2, 0.3]));
    let out = run_prog(&mut env, "argtop_k(v, 0)");
    assert_eq!(out.shape().dims(), &[0]);
    assert!(out.data().is_empty());
}
