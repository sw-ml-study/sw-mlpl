use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

// -- Scalar literals --

#[test]
fn eval_int_scalar() {
    let arr = eval("42").unwrap();
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[42.0]);
}

#[test]
fn eval_float_scalar() {
    let arr = eval("1.5").unwrap();
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[1.5]);
}

#[test]
fn eval_negative_scalar() {
    let arr = eval("-3").unwrap();
    assert_eq!(arr.data(), &[-3.0]);
}

// -- Array literals --

#[test]
fn eval_flat_array() {
    let arr = eval("[1, 2, 3]").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn eval_nested_array() {
    let arr = eval("[[1, 2], [3, 4]]").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn eval_nested_2x3() {
    let arr = eval("[[1, 2, 3], [4, 5, 6]]").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// -- Variables --

#[test]
fn eval_assign_and_lookup() {
    let tokens = lex("x = 5\nx").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let result = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(result.data(), &[5.0]);
}

#[test]
fn eval_assign_array_and_lookup() {
    let tokens = lex("x = [1, 2, 3]\nx").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let result = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(result.shape(), &Shape::vector(3));
    assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn eval_undefined_variable() {
    let result = eval("x");
    assert_eq!(result, Err(EvalError::UndefinedVariable("x".into())));
}

// -- Empty input --

#[test]
fn eval_empty() {
    let result = eval("");
    assert_eq!(result, Err(EvalError::EmptyInput));
}

// -- Unsupported (BinOp and FnCall deferred) --

#[test]
fn eval_binop_unsupported() {
    let result = eval("1 + 2");
    assert!(matches!(result, Err(EvalError::Unsupported(_))));
}

#[test]
fn eval_fncall_unsupported() {
    let result = eval("iota(5)");
    assert!(matches!(result, Err(EvalError::Unsupported(_))));
}

// -- Assignment returns value --

#[test]
fn eval_assign_returns_value() {
    let arr = eval("x = 42").unwrap();
    assert_eq!(arr.data(), &[42.0]);
}
