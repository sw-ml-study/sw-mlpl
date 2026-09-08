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
fn length_one_array_broadcasts_like_a_scalar() {
    // NumPy / APL semantics: a length-1 operand broadcasts against the
    // other shape (../emufpga request), so an indexing result that
    // comes back as `[x]` meets a vector without a reshape collapse.
    assert_eq!(eval("[2] * [1, 2, 3]").unwrap().data(), &[2.0, 4.0, 6.0]);
    assert_eq!(
        eval("[1, 2, 3] + [10]").unwrap().data(),
        &[11.0, 12.0, 13.0]
    );
    // A genuine multi-element shape mismatch is still an error.
    assert!(eval("[1, 2] + [1, 2, 3]").is_err());
}

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

// -- Regression: all-unit shapes survive scalar broadcast (BUG 1,
//    demo-abstract-algebra sw-mlpl-bug-report.md). A rank-0 scalar
//    broadcast against an array whose every axis has extent 1 used to
//    collapse the result to rank 0 ([[0]] * 1 -> [] instead of [1, 1]).

#[test]
fn scalar_broadcast_preserves_all_unit_rank() {
    assert_eq!(eval("[0] * 1").unwrap().shape(), &Shape::new(vec![1]));
    assert_eq!(eval("[[0]] * 1").unwrap().shape(), &Shape::new(vec![1, 1]));
    assert_eq!(
        eval("[[[0]]] * 1").unwrap().shape(),
        &Shape::new(vec![1, 1, 1])
    );
    // eq() broadcasting a scalar hits the same path.
    assert_eq!(
        eval("eq([[0]], 0)").unwrap().shape(),
        &Shape::new(vec![1, 1])
    );
    // A unit axis mixed with an extent > 1 axis was always fine.
    assert_eq!(
        eval("[[0, 1]] * 1").unwrap().shape(),
        &Shape::new(vec![1, 2])
    );
}
