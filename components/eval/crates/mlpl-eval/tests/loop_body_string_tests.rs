//! A string-valued statement inside a `repeat` / `train` / `for` body is
//! evaluated and discarded like any other non-final statement (microgpt-mlpl
//! bug j: the loops coerced EVERY statement's value to an array). Only the
//! value a loop consumes -- `train`'s per-step loss, `for`'s per-row capture --
//! must be an array, and that error names the construct.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let mut env = Environment::new();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), &mut env)?;
    }
    Ok(last)
}

fn ok(src: &str) {
    if let Err(e) = eval(src) {
        panic!("{src}: {e}");
    }
}

#[test]
fn string_statements_inside_loop_bodies_are_fine() {
    ok("repeat 1 { q = \"abc\"; 0 }");
    ok("repeat 1 { print(\"abc\"); 0 }");
    ok("train 1 { q = \"abc\"; 0 }");
    ok("for r in [1, 2] { q = \"abc\"; 0 }");
    ok("def u:f() { repeat 1 { q = \"abc\"; 0 }; 1 }\nu:f()");
}

#[test]
fn repeat_tolerates_a_string_final_statement() {
    ok("repeat 2 { print(\"abc\") }");
    ok("repeat 2 { q = \"abc\" }");
}

#[test]
fn loop_values_are_unchanged() {
    let losses = match eval("train 3 { q = \"x\"; step * 2 }\nlast_losses").unwrap() {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("{other:?}"),
    };
    assert_eq!(losses, vec![0.0, 2.0, 4.0]);
    let rows = match eval("for r in [1, 2] { q = \"x\"; r * 10 }\nlast_rows").unwrap() {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("{other:?}"),
    };
    assert_eq!(rows, vec![10.0, 20.0]);
}

#[test]
fn a_consumed_string_value_names_the_construct() {
    let msg = eval("train 1 { \"abc\" }").unwrap_err().to_string();
    assert!(msg.contains("train") && msg.contains("string"), "{msg}");
    let msg = eval("for r in [1, 2] { \"abc\" }").unwrap_err().to_string();
    assert!(msg.contains("for") && msg.contains("string"), "{msg}");
}
