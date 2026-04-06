//! End-to-end tests: source string -> lex -> parse -> eval -> result.

use mlpl_array::Shape;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

fn run_with_env(src: &str, env: &mut Environment) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

// -- Syntax-core-v1 examples --

#[test]
fn e2e_scalar_arithmetic() {
    assert_eq!(run("1 + 2").data(), &[3.0]);
}

#[test]
fn e2e_vector_arithmetic() {
    let arr = run("[1, 2, 3] + [4, 5, 6]");
    assert_eq!(arr.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn e2e_scalar_broadcast() {
    let arr = run("[1, 2, 3] * 10");
    assert_eq!(arr.data(), &[10.0, 20.0, 30.0]);
}

#[test]
fn e2e_iota_reshape() {
    let mut env = Environment::new();
    run_with_env("x = iota(12)", &mut env);
    let m = run_with_env("m = reshape(x, [3, 4])", &mut env);
    assert_eq!(m.shape(), &Shape::new(vec![3, 4]));
    assert_eq!(
        m.data(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn e2e_transpose() {
    let mut env = Environment::new();
    run_with_env("x = iota(12)", &mut env);
    run_with_env("m = reshape(x, [3, 4])", &mut env);
    let t = run_with_env("t = transpose(m)", &mut env);
    assert_eq!(t.shape(), &Shape::new(vec![4, 3]));
}

#[test]
fn e2e_shape_query() {
    let mut env = Environment::new();
    run_with_env("x = iota(12)", &mut env);
    run_with_env("m = reshape(x, [3, 4])", &mut env);
    run_with_env("t = transpose(m)", &mut env);
    let s = run_with_env("shape(t)", &mut env);
    assert_eq!(s.data(), &[4.0, 3.0]);
}

#[test]
fn e2e_reduce() {
    assert_eq!(run("reduce_add([1, 2, 3, 4, 5])").data(), &[15.0]);
}

#[test]
fn e2e_multi_step_computation() {
    let mut env = Environment::new();
    run_with_env("data = [1, 2, 3, 4, 5, 6]", &mut env);
    run_with_env("grid = reshape(data, [2, 3])", &mut env);
    run_with_env("scaled = grid * 2", &mut env);
    let result = run_with_env("result = reduce_add(scaled)", &mut env);
    assert_eq!(result.data(), &[42.0]);
}
