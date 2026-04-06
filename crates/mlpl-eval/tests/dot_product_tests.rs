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
fn dot_basic() {
    let arr = eval("dot([1, 2, 3], [4, 5, 6])").unwrap();
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[32.0]);
}

#[test]
fn dot_orthogonal() {
    let arr = eval("dot([1, 0], [0, 1])").unwrap();
    assert_eq!(arr.data(), &[0.0]);
}

#[test]
fn dot_length_one() {
    let arr = eval("dot([2], [3])").unwrap();
    assert_eq!(arr.data(), &[6.0]);
}

#[test]
fn dot_length_mismatch() {
    let result = eval("dot([1, 2], [1, 2, 3])");
    assert!(result.is_err());
}

#[test]
fn dot_rank_mismatch() {
    let result = eval("dot([[1, 2], [3, 4]], [1, 2])");
    assert!(result.is_err());
}
