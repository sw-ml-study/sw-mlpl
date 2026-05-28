use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

#[test]
fn exp_zero() {
    let arr = eval("exp(0)").unwrap();
    assert!((arr.data()[0] - 1.0).abs() < 1e-10);
}

#[test]
fn exp_vector() {
    let arr = eval("exp([0, 1])").unwrap();
    assert!((arr.data()[0] - 1.0).abs() < 1e-10);
    assert!((arr.data()[1] - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn log_one() {
    let arr = eval("log(1)").unwrap();
    assert!((arr.data()[0]).abs() < 1e-10);
}

#[test]
fn log_exp_roundtrip() {
    let arr = eval("log(exp(2))").unwrap();
    assert!((arr.data()[0] - 2.0).abs() < 1e-10);
}

#[test]
fn sqrt_scalar() {
    let arr = eval("sqrt(4)").unwrap();
    assert!((arr.data()[0] - 2.0).abs() < 1e-10);
}

#[test]
fn sqrt_vector() {
    let arr = eval("sqrt([1, 4, 9])").unwrap();
    assert!((arr.data()[0] - 1.0).abs() < 1e-10);
    assert!((arr.data()[1] - 2.0).abs() < 1e-10);
    assert!((arr.data()[2] - 3.0).abs() < 1e-10);
}

#[test]
fn abs_scalar() {
    let arr = eval("abs(-5)").unwrap();
    assert_eq!(arr.data(), &[5.0]);
}

#[test]
fn abs_vector() {
    let arr = eval("abs([-3, 0, 3])").unwrap();
    assert_eq!(arr.data(), &[3.0, 0.0, 3.0]);
}
