//! RS5 (../reasoning-from-scratch): scientific-notation literals evaluate
//! end to end (lexer -> parser -> eval).

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

#[test]
fn scientific_literals_evaluate() {
    for (src, want) in [
        ("1e-4", 1e-4),
        ("1.5e3", 1.5e3),
        ("2E-10", 2E-10),
        ("6.02e23", 6.02e23),
        ("1e10", 1e10),
    ] {
        let v = eval(src).unwrap().data()[0];
        assert!(
            (v - want).abs() <= want.abs() * 1e-9 + 1e-18,
            "{src}: got {v}"
        );
    }
}

#[test]
fn scientific_literal_in_expression() {
    // A learning-rate-style use: 1 - 1e-1 = 0.9.
    let v = eval("1 - 1e-1").unwrap().data()[0];
    assert!((v - 0.9).abs() < 1e-12, "got {v}");
}

#[test]
fn plain_numbers_unchanged() {
    assert!((eval("42").unwrap().data()[0] - 42.0).abs() < 1e-12);
    assert!((eval("3.5").unwrap().data()[0] - 3.5).abs() < 1e-12);
    // range(n) (the array constructor) is unaffected by exponent lexing.
    assert_eq!(eval("range(3)").unwrap().data().len(), 3);
}
