use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

fn eval_with_env(src: &str, env: &mut Environment) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env)
}

#[test]
fn repeat_increment() {
    // eval_n(3, "x = x + 1") with x=0 -> x is 3
    let mut env = Environment::new();
    eval_with_env("x = 0", &mut env).unwrap();
    let result = eval_with_env("repeat 3 { x = x + 1 }\nx", &mut env).unwrap();
    assert_eq!(result.data(), &[3.0]);
}

#[test]
fn repeat_decrement() {
    // eval_n(10, "w = w - 0.1") with w=1 -> w is ~0.0
    let mut env = Environment::new();
    eval_with_env("w = 1.0", &mut env).unwrap();
    eval_with_env("repeat 10 { w = w - 0.1 }", &mut env).unwrap();
    let result = eval_with_env("w", &mut env).unwrap();
    assert!(result.data()[0].abs() < 1e-10);
}

#[test]
fn repeat_zero_noop() {
    // eval_n(0, "x = 1") -> no-op
    let result = eval("x = 42\nrepeat 0 { x = 1 }\nx").unwrap();
    assert_eq!(result.data(), &[42.0]);
}

#[test]
fn repeat_returns_last_body_result() {
    let result = eval("x = 0\nrepeat 5 { x = x + 1 }").unwrap();
    // Last body evaluation: x = 5
    assert_eq!(result.data(), &[5.0]);
}

#[test]
fn repeat_multi_statement_body() {
    let mut env = Environment::new();
    eval_with_env("x = 0\ny = 10", &mut env).unwrap();
    eval_with_env("repeat 3 { x = x + 1; y = y - 1 }", &mut env).unwrap();
    let x = eval_with_env("x", &mut env).unwrap();
    let y = eval_with_env("y", &mut env).unwrap();
    assert_eq!(x.data(), &[3.0]);
    assert_eq!(y.data(), &[7.0]);
}

#[test]
fn repeat_with_array_ops() {
    let result = eval("v = [1, 2, 3]\nrepeat 3 { v = v + [1, 1, 1] }\nv").unwrap();
    assert_eq!(result.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn repeat_zero_returns_zero_scalar() {
    // When repeat runs 0 times, it returns the default scalar 0
    let result = eval("repeat 0 { 42 }").unwrap();
    assert_eq!(result.data(), &[0.0]);
}

#[test]
fn repeat_single_iteration() {
    let result = eval("x = 0\nrepeat 1 { x = x + 10 }\nx").unwrap();
    assert_eq!(result.data(), &[10.0]);
}
