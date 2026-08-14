//! String usability fixes (user report 2026-08-13): a string
//! assignment echoes the string, and `disp` accepts a string.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Value {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program_value(&stmts, &mut Environment::new()).unwrap()
}

#[test]
fn string_assignment_echoes_the_string() {
    // `x = "foo"` now yields "foo", like `x = [1,2,3]` yields the array
    // (not a placeholder 0).
    assert_eq!(eval("x = \"foo\""), Value::Str("foo".into()));
}

#[test]
fn disp_of_a_string_returns_the_string() {
    assert_eq!(eval("disp(\"foo\")"), Value::Str("foo".into()));
}

#[test]
fn disp_of_an_array_still_boxes_without_error() {
    match eval("disp([1, 2, 3])") {
        Value::Str(_) => {}
        other => panic!("expected a boxed string, got {other:?}"),
    }
}
