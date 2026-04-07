use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

// -- gt --

#[test]
fn gt_vector_vector() {
    let arr = eval("gt([3, 1, 2], [2, 2, 2])").unwrap();
    assert_eq!(arr.data(), &[1.0, 0.0, 0.0]);
}

#[test]
fn gt_scalar_broadcast() {
    let arr = eval("gt([1, 2, 3], 2)").unwrap();
    assert_eq!(arr.data(), &[0.0, 0.0, 1.0]);
}

#[test]
fn gt_scalars() {
    let arr = eval("gt(5, 3)").unwrap();
    assert_eq!(arr.data(), &[1.0]);
}

#[test]
fn gt_equal_values() {
    let arr = eval("gt(2, 2)").unwrap();
    assert_eq!(arr.data(), &[0.0]);
}

// -- lt --

#[test]
fn lt_scalars() {
    let arr = eval("lt(1, 2)").unwrap();
    assert_eq!(arr.data(), &[1.0]);
}

#[test]
fn lt_vector() {
    let arr = eval("lt([3, 1, 2], [2, 2, 2])").unwrap();
    assert_eq!(arr.data(), &[0.0, 1.0, 0.0]);
}

#[test]
fn lt_scalar_broadcast() {
    let arr = eval("lt([1, 2, 3], 2)").unwrap();
    assert_eq!(arr.data(), &[1.0, 0.0, 0.0]);
}

// -- eq --

#[test]
fn eq_vector_vector() {
    let arr = eval("eq([1, 2, 3], [1, 0, 3])").unwrap();
    assert_eq!(arr.data(), &[1.0, 0.0, 1.0]);
}

#[test]
fn eq_scalars() {
    let arr = eval("eq(5, 5)").unwrap();
    assert_eq!(arr.data(), &[1.0]);
}

#[test]
fn eq_scalar_broadcast() {
    let arr = eval("eq([1, 2, 2, 3], 2)").unwrap();
    assert_eq!(arr.data(), &[0.0, 1.0, 1.0, 0.0]);
}

// -- mean --

#[test]
fn mean_vector() {
    let arr = eval("mean([2, 4, 6])").unwrap();
    assert_eq!(arr.rank(), 0);
    assert!((arr.data()[0] - 4.0).abs() < 1e-10);
}

#[test]
fn mean_matrix() {
    let arr = eval("mean(reshape(iota(6), [2, 3]))").unwrap();
    assert_eq!(arr.rank(), 0);
    assert!((arr.data()[0] - 2.5).abs() < 1e-10);
}

#[test]
fn mean_scalar() {
    let arr = eval("mean(42)").unwrap();
    assert_eq!(arr.data(), &[42.0]);
}

#[test]
fn mean_single_element() {
    let arr = eval("mean([7])").unwrap();
    assert!((arr.data()[0] - 7.0).abs() < 1e-10);
}

// -- combined usage --

#[test]
fn accuracy_pattern() {
    // Simulate accuracy: mean of correct predictions
    let arr = eval("mean(eq([1, 0, 1, 1], [1, 1, 1, 0]))").unwrap();
    assert!((arr.data()[0] - 0.5).abs() < 1e-10);
}
