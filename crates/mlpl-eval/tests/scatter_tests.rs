//! Saga 20 step 003: `scatter(buffer, index, value)`.
//!
//! Rank-1 scalar scatter -- returns a copy of `buffer` with the
//! single entry at `index` replaced by `value`. Pairs with
//! `repeat N { ... }` loops that produce one number per iteration
//! and need to accumulate them into a rank-1 accumulator.

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
fn scatter_writes_single_entry_into_zero_buffer() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![4], vec![0.0, 0.0, 0.0, 0.0]));
    let out = run_prog(&mut env, "scatter(b, 2, 7.5)");
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.data(), &[0.0, 0.0, 7.5, 0.0]);
}

#[test]
fn scatter_overwrites_existing_entry_at_index() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![3], vec![1.0, 2.0, 3.0]));
    let out = run_prog(&mut env, "scatter(b, 0, 99.0)");
    assert_eq!(out.data(), &[99.0, 2.0, 3.0]);
}

#[test]
fn scatter_end_index_is_valid() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![3], vec![1.0, 2.0, 3.0]));
    let out = run_prog(&mut env, "scatter(b, 2, 5.0)");
    assert_eq!(out.data(), &[1.0, 2.0, 5.0]);
}

#[test]
fn scatter_is_not_in_place_at_source_level() {
    // The source binding is preserved; the new tensor is a copy.
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![3], vec![1.0, 2.0, 3.0]));
    let _ = run_prog(&mut env, "scatter(b, 0, 99.0)");
    assert_eq!(
        env.get("b").unwrap().data(),
        &[1.0, 2.0, 3.0],
        "scatter must not mutate the source binding"
    );
}

#[test]
fn scatter_rejects_negative_index() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![4], vec![0.0; 4]));
    let stmts = parse(&lex("scatter(b, -1, 1.0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("negative index should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter") && (msg.contains("index") || msg.contains("bounds")),
        "error should mention scatter and index/bounds, got: {msg}"
    );
}

#[test]
fn scatter_rejects_index_equal_to_len() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![4], vec![0.0; 4]));
    let stmts = parse(&lex("scatter(b, 4, 1.0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("index=len should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter") && (msg.contains("index") || msg.contains("bounds")),
        "error should mention scatter and index/bounds, got: {msg}"
    );
}

#[test]
fn scatter_rejects_non_rank_1_buffer() {
    let mut env = Environment::new();
    env.set("m".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let stmts = parse(&lex("scatter(m, 0, 9.0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("non-rank-1 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter") && (msg.contains("rank") || msg.contains("vector")),
        "error should mention rank/vector, got: {msg}"
    );
}

#[test]
fn scatter_rejects_non_scalar_index() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![4], vec![0.0; 4]));
    env.set("i".into(), arr(vec![2], vec![0.0, 1.0]));
    let stmts = parse(&lex("scatter(b, i, 9.0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("non-scalar index should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter") && (msg.contains("scalar") || msg.contains("index")),
        "error should mention scatter and scalar/index, got: {msg}"
    );
}

#[test]
fn scatter_rejects_wrong_arity() {
    let mut env = Environment::new();
    env.set("b".into(), arr(vec![4], vec![0.0; 4]));
    let stmts = parse(&lex("scatter(b, 2)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("2 args should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter"),
        "error should mention scatter, got: {msg}"
    );
}

#[test]
fn scatter_chained_writes_accumulate() {
    // The fundamental compositional property: a sequence of scatter
    // calls on the threaded output builds up a fully-populated
    // buffer. Step 004's demo will drive this via a loop construct
    // (for / repeat); step 003 just proves the composition itself.
    let mut env = Environment::new();
    env.set("buf".into(), arr(vec![4], vec![0.0; 4]));
    run_prog(&mut env, "buf = scatter(buf, 0, 1.0)");
    run_prog(&mut env, "buf = scatter(buf, 1, 2.0)");
    run_prog(&mut env, "buf = scatter(buf, 2, 3.0)");
    run_prog(&mut env, "buf = scatter(buf, 3, 4.0)");
    let buf = env.get("buf").expect("buf bound");
    assert_eq!(buf.data(), &[1.0, 2.0, 3.0, 4.0]);
}
