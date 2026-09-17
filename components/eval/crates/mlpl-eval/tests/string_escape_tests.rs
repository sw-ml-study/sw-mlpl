//! RS-escape (../reasoning-from-scratch): LaTeX/math text in a string
//! literal lexes end to end (unknown backslash escapes stay literal).

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval_str(src: &str) -> Result<DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

#[test]
fn latex_string_length_is_literal() {
    // "\frac" is the 5 chars backslash,f,r,a,c -- str_len counts them.
    assert!((eval_str(r#"str_len("\frac")"#).unwrap().data()[0] - 5.0).abs() < 1e-9);
    // "\in" is 3 chars.
    assert!((eval_str(r#"str_len("\in")"#).unwrap().data()[0] - 3.0).abs() < 1e-9);
}

#[test]
fn known_escapes_still_control_chars() {
    // "\n" is a single newline character.
    assert!((eval_str(r#"str_len("\n")"#).unwrap().data()[0] - 1.0).abs() < 1e-9);
}
