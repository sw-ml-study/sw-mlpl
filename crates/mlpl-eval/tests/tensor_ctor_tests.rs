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
fn param_vector_zeros() {
    let arr = eval("param[3]").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn param_matrix_zeros() {
    let arr = eval("param[2, 3]").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0; 6]);
}

#[test]
fn tensor_matrix_zeros() {
    let arr = eval("tensor[4, 5]").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![4, 5]));
    assert_eq!(arr.data(), &[0.0; 20]);
}

#[test]
fn tensor_rank3_zeros() {
    let arr = eval("tensor[2, 3, 4]").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3, 4]));
    assert_eq!(arr.data(), &[0.0; 24]);
}

#[test]
fn param_assign_and_use() {
    // Assigning to a param ctor gives a regular array that the rest
    // of the language can continue to use as before.
    let tokens = lex("w = param[2, 2]\nw + ones([2, 2])").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let arr = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(arr.data(), &[1.0, 1.0, 1.0, 1.0]);
}
