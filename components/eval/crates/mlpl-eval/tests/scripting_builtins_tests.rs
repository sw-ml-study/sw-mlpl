//! `to_number(s)`, `to_int(s)`, `env(name)` builtins. Saga 31 step 002.
//!
//! All three return `Value::Result` so the caller is forced to
//! handle success vs failure explicitly via `is_ok` / `unwrap_or`
//! / `err_message`. The conversion behavior:
//!
//! - `to_number("42")` -> `Ok(42)`, `to_number("4.25")` -> `Ok(4.25)`
//! - `to_number("abc")` -> `Err("to_number: cannot parse \"abc\" ...")`
//! - `to_int("42")` -> `Ok(42)`, `to_int("3.5")` -> `Err(...)` (integer-only)
//! - `env("SET_NAME")` -> `Ok("value")`, `env("UNSET_NAME")` -> `Err(...)`

use mlpl_eval::{Environment, Value, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn val(src: &str, env: &mut Environment) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn err_msg_of(src: &str, env: &mut Environment) -> String {
    match val(src, env) {
        Value::Result { ok: false, payload } => format!("{payload}"),
        other => panic!("expected Err, got {other}"),
    }
}

fn scalar(v: Value) -> f64 {
    match v {
        Value::Array(a) => {
            assert_eq!(a.shape().dims(), [] as [usize; 0]);
            a.data()[0]
        }
        other => panic!("expected scalar Array, got {other}"),
    }
}

fn str_of(v: Value) -> String {
    match v {
        Value::Str(s) => s,
        other => panic!("expected Str, got {other}"),
    }
}

#[test]
fn to_number_ok_on_integer_literal() {
    let mut env = Environment::new();
    let v = val("unwrap(to_number(\"42\"))", &mut env);
    assert_eq!(scalar(v), 42.0);
}

#[test]
fn to_number_ok_on_float_literal() {
    let mut env = Environment::new();
    let v = val("unwrap(to_number(\"4.25\"))", &mut env);
    assert!((scalar(v) - 4.25).abs() < 1e-9);
}

#[test]
fn to_number_ok_handles_negative_and_trim() {
    let mut env = Environment::new();
    let v1 = val("unwrap(to_number(\"-7.5\"))", &mut env);
    assert!((scalar(v1) - -7.5).abs() < 1e-9);
    let v2 = val("unwrap(to_number(\"  42  \"))", &mut env);
    assert_eq!(scalar(v2), 42.0);
}

#[test]
fn to_number_err_on_non_numeric() {
    let mut env = Environment::new();
    // is_ok returns 0.0 for Err
    let v = val("is_ok(to_number(\"abc\"))", &mut env);
    assert_eq!(scalar(v), 0.0);
    let msg = err_msg_of("to_number(\"abc\")", &mut env);
    assert!(
        msg.contains("to_number") && msg.contains("abc"),
        "expected err to mention to_number + abc, got: {msg}"
    );
}

#[test]
fn to_int_ok_on_integer_literal() {
    let mut env = Environment::new();
    let v = val("unwrap(to_int(\"42\"))", &mut env);
    assert_eq!(scalar(v), 42.0);
}

#[test]
fn to_int_err_on_float_literal() {
    let mut env = Environment::new();
    let v = val("is_ok(to_int(\"3.5\"))", &mut env);
    assert_eq!(scalar(v), 0.0);
    let msg = err_msg_of("to_int(\"3.5\")", &mut env);
    assert!(
        msg.contains("to_int") && msg.contains("not an integer"),
        "expected err to mention 'not an integer', got: {msg}"
    );
}

#[test]
fn to_int_err_on_non_numeric() {
    let mut env = Environment::new();
    let msg = err_msg_of("to_int(\"xyz\")", &mut env);
    assert!(
        msg.contains("to_int") && msg.contains("xyz"),
        "expected err to mention to_int + xyz, got: {msg}"
    );
}

#[test]
fn env_ok_when_set() {
    let var = "MLPL_TEST_ENV_OK_31_002";
    // SAFETY: setting a process-wide env var. Each test uses a
    // unique var name so the only race is read-back of our own write.
    unsafe { std::env::set_var(var, "hello-world") };
    let mut env = Environment::new();
    let src = format!("unwrap(env(\"{var}\"))");
    let v = val(&src, &mut env);
    assert_eq!(str_of(v), "hello-world");
    unsafe { std::env::remove_var(var) };
}

#[test]
fn env_err_when_not_set() {
    let var = "MLPL_TEST_ENV_NEVER_SET_31_002";
    unsafe { std::env::remove_var(var) };
    let mut env = Environment::new();
    let v = val(&format!("is_ok(env(\"{var}\"))"), &mut env);
    assert_eq!(scalar(v), 0.0);
    let msg = err_msg_of(&format!("env(\"{var}\")"), &mut env);
    assert!(
        msg.contains("env") && msg.contains(var) && msg.contains("not set"),
        "expected err to mention env, var name, 'not set'; got: {msg}"
    );
}

#[test]
fn rejects_non_string_argument() {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex("to_number(42)").unwrap()).unwrap(), &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("to_number") && msg.contains("string"),
        "expected non-string error, got: {msg}"
    );
}

#[test]
fn rejects_wrong_arity() {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex("to_number()").unwrap()).unwrap(), &mut env).unwrap_err();
    assert!(format!("{err}").contains("to_number"));
    let err2 = eval_program(
        &parse(&lex("env(\"A\", \"B\")").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap_err();
    assert!(format!("{err2}").contains("env"));
}

#[test]
fn unwrap_or_default_path() {
    // Canonical scripting pattern: read an env var with a default
    // fallback in one expression.
    let var = "MLPL_TEST_ENV_NEVER_SET_31_002_UO";
    unsafe { std::env::remove_var(var) };
    let mut env = Environment::new();
    let v = val(
        &format!("unwrap_or(env(\"{var}\"), \"default-path\")"),
        &mut env,
    );
    assert_eq!(str_of(v), "default-path");
}
