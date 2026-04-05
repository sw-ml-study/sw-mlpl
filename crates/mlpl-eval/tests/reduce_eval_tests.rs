use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

#[test]
fn reduce_add_vector() {
    let arr = eval("reduce_add([1, 2, 3])").unwrap();
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[6.0]);
}

#[test]
fn reduce_mul_vector() {
    let arr = eval("reduce_mul([1, 2, 3, 4])").unwrap();
    assert_eq!(arr.data(), &[24.0]);
}

#[test]
fn reduce_add_iota() {
    let arr = eval("reduce_add(iota(5))").unwrap();
    assert_eq!(arr.data(), &[10.0]);
}

#[test]
fn reduce_add_matrix() {
    let arr = eval("reduce_add(reshape(iota(6), [2, 3]))").unwrap();
    assert_eq!(arr.data(), &[15.0]);
}

#[test]
fn reduce_add_scalar() {
    let arr = eval("reduce_add(42)").unwrap();
    assert_eq!(arr.data(), &[42.0]);
}

#[test]
fn reduce_mul_with_zero() {
    let arr = eval("reduce_mul([2, 0, 3])").unwrap();
    assert_eq!(arr.data(), &[0.0]);
}
