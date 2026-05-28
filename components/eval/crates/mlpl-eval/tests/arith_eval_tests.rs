use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

// -- Scalar arithmetic --

#[test]
fn scalar_add() {
    let arr = eval("1 + 2").unwrap();
    assert_eq!(arr.data(), &[3.0]);
}

#[test]
fn scalar_sub() {
    let arr = eval("5 - 3").unwrap();
    assert_eq!(arr.data(), &[2.0]);
}

#[test]
fn scalar_mul() {
    let arr = eval("3 * 4").unwrap();
    assert_eq!(arr.data(), &[12.0]);
}

#[test]
fn scalar_div() {
    let arr = eval("10 / 4").unwrap();
    assert_eq!(arr.data(), &[2.5]);
}

// -- Precedence --

#[test]
fn precedence_mul_add() {
    let arr = eval("1 + 2 * 3").unwrap();
    assert_eq!(arr.data(), &[7.0]);
}

#[test]
fn precedence_parens() {
    let arr = eval("(1 + 2) * 3").unwrap();
    assert_eq!(arr.data(), &[9.0]);
}

// -- Element-wise vector --

#[test]
fn vector_add() {
    let arr = eval("[1, 2, 3] + [4, 5, 6]").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn vector_sub() {
    let arr = eval("[10, 20, 30] - [1, 2, 3]").unwrap();
    assert_eq!(arr.data(), &[9.0, 18.0, 27.0]);
}

// -- Scalar broadcasting --

#[test]
fn broadcast_scalar_times_vector() {
    let arr = eval("[1, 2, 3] * 10").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[10.0, 20.0, 30.0]);
}

#[test]
fn broadcast_vector_times_scalar() {
    let arr = eval("10 * [1, 2, 3]").unwrap();
    assert_eq!(arr.data(), &[10.0, 20.0, 30.0]);
}

#[test]
fn broadcast_scalar_add_vector() {
    let arr = eval("[1, 2, 3] + 1").unwrap();
    assert_eq!(arr.data(), &[2.0, 3.0, 4.0]);
}

// -- Shape mismatch --

#[test]
fn shape_mismatch() {
    let result = eval("[1, 2] + [1, 2, 3]");
    assert!(
        matches!(result, Err(EvalError::ShapeMismatch { ref op, .. }) if op == "add"),
        "got {result:?}"
    );
}

// -- With variables --

#[test]
fn variable_arithmetic() {
    let tokens = lex("x = [1, 2, 3]\nx + 1").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let result = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
}

// -- Division by zero (IEEE) --

#[test]
fn div_by_zero_inf() {
    let arr = eval("1 / 0").unwrap();
    assert!(arr.data()[0].is_infinite());
}
