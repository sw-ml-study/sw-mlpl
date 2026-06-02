//! Issue #6 / C3: a bare `if cond { body }` (no else) evaluates to the
//! body when taken and to `0` when not.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> f64 {
    let mut env = Environment::new();
    eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env)
        .unwrap()
        .data()[0]
}

#[test]
fn bare_if_taken_yields_body() {
    assert_eq!(eval("if gt(3, 0) { 42 }"), 42.0);
}

#[test]
fn bare_if_not_taken_yields_zero() {
    assert_eq!(eval("if gt(0, 1) { 42 }"), 0.0);
}

#[test]
fn bare_if_as_guard_then_continue() {
    // Statement-position bare-if for a side effect, then more code.
    assert_eq!(eval("x = 1\nif gt(5, 0) { x = 9 }\nx"), 9.0);
}
