use mlpl_array::Shape;
use mlpl_eval::{EvalError, evaluate};
use mlpl_parser::lex;

#[test]
fn eval_number_sequence() {
    let tokens = lex("1 2 3").unwrap();
    let result = evaluate(&tokens).unwrap();
    assert_eq!(result.shape(), &Shape::vector(3));
    assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn eval_single_number() {
    let tokens = lex("42").unwrap();
    let result = evaluate(&tokens).unwrap();
    assert_eq!(result.shape(), &Shape::scalar());
    assert_eq!(result.data(), &[42.0]);
}

#[test]
fn eval_floats() {
    let tokens = lex("1.5 2.5").unwrap();
    let result = evaluate(&tokens).unwrap();
    assert_eq!(result.shape(), &Shape::vector(2));
    assert_eq!(result.data(), &[1.5, 2.5]);
}

#[test]
fn eval_negative_numbers() {
    let tokens = lex("-1 0 1").unwrap();
    let result = evaluate(&tokens).unwrap();
    assert_eq!(result.shape(), &Shape::vector(3));
    assert_eq!(result.data(), &[-1.0, 0.0, 1.0]);
}

#[test]
fn eval_empty_input() {
    let tokens = lex("").unwrap();
    let result = evaluate(&tokens);
    assert_eq!(result, Err(EvalError::EmptyInput));
}

#[test]
fn eval_comment_only() {
    let tokens = lex("# nothing here").unwrap();
    let result = evaluate(&tokens);
    assert_eq!(result, Err(EvalError::EmptyInput));
}

#[test]
fn eval_unsupported_token() {
    let tokens = lex("x = 1").unwrap();
    let result = evaluate(&tokens);
    assert!(matches!(result, Err(EvalError::Unsupported(_))));
}
