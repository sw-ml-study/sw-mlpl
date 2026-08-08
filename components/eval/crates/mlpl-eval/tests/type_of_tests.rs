//! `type_of(v)` -- expose the internal value-kind string as a
//! builtin so a program can branch on a value's kind at the
//! root (demo-algorithms non-record root type detection). Total:
//! works on any value, never errors.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn kind_of(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Str(s) => s,
        other => panic!("type_of should return a string from {src}, got {other:?}"),
    }
}

#[test]
fn type_of_array_and_scalar() {
    let mut env = Environment::new();
    assert_eq!(kind_of(&mut env, "type_of([1, 2, 3])"), "array");
    // a bare number is a rank-0 array
    assert_eq!(kind_of(&mut env, "type_of(7)"), "array");
}

#[test]
fn type_of_string() {
    let mut env = Environment::new();
    assert_eq!(kind_of(&mut env, "type_of(\"hi\")"), "string");
}

#[test]
fn type_of_record() {
    let mut env = Environment::new();
    assert_eq!(kind_of(&mut env, "type_of({a: 1, b: 2})"), "record");
}

#[test]
fn type_of_result() {
    let mut env = Environment::new();
    assert_eq!(
        kind_of(&mut env, "type_of(record_get({a: 1}, \"a\"))"),
        "result"
    );
    assert_eq!(
        kind_of(&mut env, "type_of(record_get({a: 1}, \"z\"))"),
        "result"
    );
}

#[test]
fn type_of_reference_kinds() {
    let mut env = Environment::new();
    // a builtin reference names itself; a user-fn reference too.
    assert_eq!(kind_of(&mut env, "type_of(:add)"), "builtin-ref");
    eval_value(&mut env, "def u:sq(x) { x * x }").unwrap();
    assert_eq!(kind_of(&mut env, "type_of(:u:sq)"), "user-fn-ref");
}

#[test]
fn type_of_is_total_on_a_partial() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:k(x, y) { x }").unwrap();
    // under-application yields a partial; type_of names it
    // rather than erroring.
    assert_eq!(kind_of(&mut env, "type_of(call(:u:k, 5))"), "partial");
}

#[test]
fn type_of_wrong_arity_errors() {
    let mut env = Environment::new();
    assert!(eval_value(&mut env, "type_of(1, 2)").is_err());
    assert!(eval_value(&mut env, "type_of()").is_err());
}
