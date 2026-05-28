use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

// -- Sigmoid --

#[test]
fn sigmoid_zero() {
    let arr = eval("sigmoid(0)").unwrap();
    assert!((arr.data()[0] - 0.5).abs() < 1e-10);
}

#[test]
fn sigmoid_large_positive() {
    let arr = eval("sigmoid(100)").unwrap();
    assert!((arr.data()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn sigmoid_large_negative() {
    let arr = eval("sigmoid(-100)").unwrap();
    assert!(arr.data()[0].abs() < 1e-10);
    assert!(!arr.data()[0].is_nan());
}

#[test]
fn sigmoid_vector() {
    let arr = eval("sigmoid([0, 100, -100])").unwrap();
    assert!((arr.data()[0] - 0.5).abs() < 1e-10);
    assert!((arr.data()[1] - 1.0).abs() < 1e-10);
    assert!(arr.data()[2].abs() < 1e-10);
}

// -- Tanh --

#[test]
fn tanh_zero() {
    let arr = eval("tanh_fn(0)").unwrap();
    assert!(arr.data()[0].abs() < 1e-10);
}

#[test]
fn tanh_vector() {
    let arr = eval("tanh_fn([0, 1, -1])").unwrap();
    assert!(arr.data()[0].abs() < 1e-10);
    assert!((arr.data()[1] - 1.0_f64.tanh()).abs() < 1e-10);
    assert!((arr.data()[2] - (-1.0_f64).tanh()).abs() < 1e-10);
}

// -- Pow --

#[test]
fn pow_scalar() {
    let arr = eval("pow(2, 3)").unwrap();
    assert_eq!(arr.data(), &[8.0]);
}

#[test]
fn pow_broadcast_scalar() {
    let arr = eval("pow([1, 2, 3], 2)").unwrap();
    assert_eq!(arr.data(), &[1.0, 4.0, 9.0]);
}

#[test]
fn pow_elementwise() {
    let arr = eval("pow([2, 3], [3, 2])").unwrap();
    assert_eq!(arr.data(), &[8.0, 9.0]);
}
