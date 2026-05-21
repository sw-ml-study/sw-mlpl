//! `if cond { then } else { else }` expression. Saga 31 step 004.
//!
//! Truthiness rules:
//! - `Value::Array` with rank 0: truthy iff non-zero
//! - `Value::Result`: truthy iff `ok` is true
//! - All other types: runtime error
//!
//! `else` is required; a parse error catches the missing-else case.

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, Value, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn val(src: &str, env: &mut Environment) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn scalar(v: Value) -> f64 {
    match v {
        Value::Array(a) => {
            assert_eq!(a.rank(), 0, "expected scalar, got rank {}", a.rank());
            a.data()[0]
        }
        other => panic!("expected scalar Array, got {other}"),
    }
}

#[test]
fn if_true_picks_then() {
    let mut env = Environment::new();
    let v = val("if 1 { 42 } else { 99 }", &mut env);
    assert_eq!(scalar(v), 42.0);
}

#[test]
fn if_false_picks_else() {
    let mut env = Environment::new();
    let v = val("if 0 { 42 } else { 99 }", &mut env);
    assert_eq!(scalar(v), 99.0);
}

#[test]
fn if_returns_last_body_expr() {
    // Multiple statements in a branch -- the value of the last one
    // is the branch value (same as repeat / train / for).
    let mut env = Environment::new();
    let v = val("if 1 { x = 10\nx * 3 } else { 0 }", &mut env);
    assert_eq!(scalar(v), 30.0);
}

#[test]
fn if_nested() {
    let mut env = Environment::new();
    let v = val("if 0 { 1 } else { if 1 { 2 } else { 3 } }", &mut env);
    assert_eq!(scalar(v), 2.0);
}

#[test]
fn if_composes_into_assignment() {
    // The canonical scripting pattern: branch on a flag, bind the result.
    let mut env = Environment::new();
    let v = val(
        "flag = 1\nx = if flag { 100 } else { 200 }\nx * 2",
        &mut env,
    );
    assert_eq!(scalar(v), 200.0);
}

#[test]
fn if_cond_uses_result_ok_flag() {
    // to_number("42") is Ok(42) -- truthy.
    let mut env = Environment::new();
    let v = val("if to_number(\"42\") { 1 } else { 2 }", &mut env);
    assert_eq!(scalar(v), 1.0);
}

#[test]
fn if_cond_uses_result_err_as_false() {
    // to_number("abc") is Err(...) -- falsy.
    let mut env = Environment::new();
    let v = val("if to_number(\"abc\") { 1 } else { 2 }", &mut env);
    assert_eq!(scalar(v), 2.0);
}

#[test]
fn if_cond_with_arithmetic() {
    let mut env = Environment::new();
    let v = val("if 3 + 2 - 5 { 1 } else { 2 }", &mut env);
    assert_eq!(scalar(v), 2.0);
}

#[test]
fn if_cond_negative_number_is_truthy() {
    // Truthy = nonzero. Negative is still truthy.
    let mut env = Environment::new();
    let v = val("if -1 { 1 } else { 2 }", &mut env);
    assert_eq!(scalar(v), 1.0);
}

#[test]
fn if_cond_non_scalar_array_errors() {
    let mut env = Environment::new();
    let err = eval_program(
        &parse(&lex("if [1, 2, 3] { 1 } else { 2 }").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("if condition"),
        "expected message about non-scalar cond, got: {msg}"
    );
}

#[test]
fn missing_else_is_parse_error() {
    let err = parse(&lex("if 1 { 42 }").unwrap()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("else") || msg.contains("Unexpected") || msg.contains("Eof"),
        "expected parse error about missing else, got: {msg}"
    );
}

#[test]
fn if_can_return_string_value() {
    // The branches can return any Value, not just scalars. Test
    // by evaluating the if expression directly (assigning to a name
    // would route the Str through env.set_string() and the assignment
    // returns a placeholder).
    let mut env = Environment::new();
    let v = val("if 1 { \"alice\" } else { \"bob\" }", &mut env);
    match v {
        Value::Str(s) => assert_eq!(s, "alice"),
        other => panic!("expected Str, got {other}"),
    }
}

#[test]
fn if_can_return_vector_value() {
    let mut env = Environment::new();
    let v = val("if 1 { [1, 2, 3] } else { [4, 5, 6] }", &mut env);
    let arr = match v {
        Value::Array(a) => a,
        other => panic!("expected Array, got {other}"),
    };
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn if_zero_scalar_array_is_falsy() {
    let mut env = Environment::new();
    let v = val("zero = 0\nif zero { 1 } else { 2 }", &mut env);
    assert_eq!(scalar(v), 2.0);
    let _ = DenseArray::from_scalar(0.0); // satisfy import-used lint
}
