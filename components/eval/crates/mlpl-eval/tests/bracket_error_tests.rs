//! bracket(setup, use, teardown) error plane: hard-error capture
//! at every hook boundary, teardown_error merging when both
//! fail, and `?` composition.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

const HOOKS: &str = "
def u:setup_plain() { {resource: 42} }
def u:setup_boom() { take([1, 2], 0, 9) }
def u:use_ok(f) { f.resource * 2 }
def u:use_err(f) { err(\"use failed\") }
def u:use_err_rec(f) { err({stage: \"use\", detail: \"bad fixture\"}) }
def u:use_boom(f) { take([1, 2], 0, 9) }
def u:teardown_ok(f) { ok(f.resource) }
def u:teardown_err(f) { err(\"teardown leaked\") }
def u:teardown_boom(f) { take([1, 2], 0, 9) }
def u:use_ok_r(f) { ok(f.resource * 2) }
";

fn err_payload(env: &mut Environment, src: &str) -> Value {
    let v = eval_value(env, &format!("is_err({src})")).unwrap();
    assert!(
        matches!(&v, Value::Array(a) if a.data()[0] == 1.0),
        "expected err from {src}"
    );
    // or_else hands the handler the error payload -- the shipped
    // programmatic route to a structured err record.
    eval_value(env, "def u:pluck(e) { e }").unwrap();
    eval_value(env, &format!("or_else(:u:pluck, {src})"))
        .unwrap_or_else(|e| panic!("payload of {src}: {e}"))
}

fn field<'a>(v: &'a Value, name: &str) -> &'a Value {
    match v {
        Value::Record { fields } => &fields[name],
        other => panic!("expected record, got {other:?}"),
    }
}

#[test]
fn hard_error_in_use_is_captured_and_teardown_still_runs() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    // A hard use error becomes the structured {kind, message}
    // record; teardown ran, PROVEN by its failure landing in
    // teardown_error next to the primary.
    let src = "bracket(:u:setup_plain, :u:use_boom, :u:teardown_err)";
    let payload = err_payload(&mut env, src);
    assert!(matches!(field(&payload, "kind"), Value::Str(s) if s == "runtime"));
    assert!(
        matches!(field(&payload, "message"), Value::Str(s) if s.contains("shape mismatch")),
        "{payload:?}"
    );
    assert!(
        matches!(field(&payload, "teardown_error"), Value::Str(s) if s == "teardown leaked"),
        "{payload:?}"
    );
}

#[test]
fn hard_error_in_teardown_after_success_is_captured() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let src = "bracket(:u:setup_plain, :u:use_ok, :u:teardown_boom)";
    let payload = err_payload(&mut env, src);
    assert!(matches!(field(&payload, "kind"), Value::Str(s) if s == "runtime"));
}

#[test]
fn both_fail_string_primary_wraps_to_message_plus_teardown_error() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let src = "bracket(:u:setup_plain, :u:use_err, :u:teardown_err)";
    let payload = err_payload(&mut env, src);
    assert!(matches!(field(&payload, "message"), Value::Str(s) if s == "use failed"));
    assert!(matches!(field(&payload, "teardown_error"), Value::Str(s) if s == "teardown leaked"));
}

#[test]
fn both_fail_record_primary_gains_teardown_error_field() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let src = "bracket(:u:setup_plain, :u:use_err_rec, :u:teardown_err)";
    let payload = err_payload(&mut env, src);
    assert!(matches!(field(&payload, "stage"), Value::Str(s) if s == "use"));
    assert!(matches!(field(&payload, "detail"), Value::Str(s) if s == "bad fixture"));
    assert!(matches!(field(&payload, "teardown_error"), Value::Str(s) if s == "teardown leaked"));
}

#[test]
fn hard_error_in_setup_skips_hooks_and_returns_the_record() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    // teardown_err would otherwise contaminate the payload.
    let src = "bracket(:u:setup_boom, :u:use_ok, :u:teardown_err)";
    let payload = err_payload(&mut env, src);
    assert!(matches!(field(&payload, "kind"), Value::Str(s) if s == "runtime"));
    assert!(
        !matches!(&payload, Value::Record { fields } if fields.contains_key("teardown_error")),
        "teardown must not have run: {payload:?}"
    );
}

#[test]
fn question_mark_composes_after_teardown() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    eval_value(
        &mut env,
        "def u:go() { x = bracket(:u:setup_plain, :u:use_err, :u:teardown_ok)?; 99 }",
    )
    .unwrap();
    let v = eval_value(&mut env, "err_message(u:go())").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "use failed"),
        "? early-returns use's failure: {v:?}"
    );
    let v = eval_value(
        &mut env,
        "def u:go_ok() { x = bracket(:u:setup_plain, :u:use_ok_r, :u:teardown_ok)?; x + 1 }\nu:go_ok()",
    )
    .unwrap();
    assert!(
        matches!(&v, Value::Array(a) if a.data()[0] == 85.0),
        "{v:?}"
    );
}
