//! Acceptance tests: verify Display output matches what the REPL shows.
//! These test the full pipeline AND the string representation the user sees.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

/// Run source through full pipeline, return Display string.
fn eval_display(src: &str, env: &mut Environment) -> String {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let arr = eval_program(&stmts, env).unwrap();
    arr.to_string()
}

// -- syntax-core-v1.md examples, verifying exact output --

#[test]
fn accept_scalar_arithmetic() {
    let mut env = Environment::new();
    assert_eq!(eval_display("1 + 2", &mut env), "3");
}

#[test]
fn accept_vector_arithmetic() {
    let mut env = Environment::new();
    assert_eq!(eval_display("[1, 2, 3] + [4, 5, 6]", &mut env), "5 7 9");
}

#[test]
fn accept_scalar_broadcast() {
    let mut env = Environment::new();
    assert_eq!(eval_display("[1, 2, 3] * 10", &mut env), "10 20 30");
}

#[test]
fn accept_iota_reshape_transpose_shape() {
    let mut env = Environment::new();
    eval_display("x = iota(12)", &mut env);
    eval_display("m = reshape(x, [3, 4])", &mut env);
    eval_display("t = transpose(m)", &mut env);
    assert_eq!(eval_display("shape(t)", &mut env), "4 3");
}

#[test]
fn accept_reduce() {
    let mut env = Environment::new();
    assert_eq!(eval_display("reduce_add([1, 2, 3, 4, 5])", &mut env), "15");
}

#[test]
fn accept_multi_step_computation() {
    let mut env = Environment::new();
    eval_display("data = [1, 2, 3, 4, 5, 6]", &mut env);
    eval_display("grid = reshape(data, [2, 3])", &mut env);
    eval_display("scaled = grid * 2", &mut env);
    assert_eq!(eval_display("result = reduce_add(scaled)", &mut env), "42");
}

#[test]
fn accept_matrix_display() {
    let mut env = Environment::new();
    eval_display("x = iota(12)", &mut env);
    let output = eval_display("m = reshape(x, [3, 4])", &mut env);
    assert_eq!(output, "0 1 2 3\n4 5 6 7\n8 9 10 11");
}

#[test]
fn accept_transpose_display() {
    let mut env = Environment::new();
    eval_display("x = iota(6)", &mut env);
    eval_display("m = reshape(x, [2, 3])", &mut env);
    let output = eval_display("transpose(m)", &mut env);
    assert_eq!(output, "0 3\n1 4\n2 5");
}

#[test]
fn accept_rank_query() {
    let mut env = Environment::new();
    assert_eq!(eval_display("rank([1, 2, 3])", &mut env), "1");
    assert_eq!(eval_display("rank(42)", &mut env), "0");
}

#[test]
fn accept_reduce_mul() {
    let mut env = Environment::new();
    assert_eq!(eval_display("reduce_mul([1, 2, 3, 4])", &mut env), "24");
}

#[test]
fn accept_nested_array_literal() {
    let mut env = Environment::new();
    let output = eval_display("[[1, 2, 3], [4, 5, 6]]", &mut env);
    assert_eq!(output, "1 2 3\n4 5 6");
}

#[test]
fn accept_variable_persistence() {
    let mut env = Environment::new();
    eval_display("x = [10, 20, 30]", &mut env);
    assert_eq!(eval_display("x + 1", &mut env), "11 21 31");
    eval_display("y = x * 2", &mut env);
    assert_eq!(eval_display("reduce_add(y)", &mut env), "120");
}
