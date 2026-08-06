//! bracket(setup, use, teardown) core sequencing: fixture flow,
//! plain-value-as-ok, setup-failure skip, teardown-after-use.
//! (Hard-error capture and diagnostic merging: bracket_error_tests.)

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(v: &Value) -> f64 {
    match v {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

const HOOKS: &str = "
def u:setup_plain() { {resource: 42} }
def u:setup_ok() { ok({resource: 42}) }
def u:setup_err() { err(\"no resource\") }
def u:use_double(f) { f.resource * 2 }
def u:use_err(f) { err(\"use failed\") }
def u:teardown_ok(f) { ok(f.resource) }
def u:teardown_err(f) { err(\"teardown leaked\") }
";

#[test]
fn fixture_flows_from_setup_to_use_and_result_is_uses() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let v = eval_value(
        &mut env,
        "bracket(:u:setup_plain, :u:use_double, :u:teardown_ok)",
    )
    .unwrap();
    assert_eq!(scalar(&v), 84.0, "plain-value setup is treated as ok");
    let v = eval_value(
        &mut env,
        "bracket(:u:setup_ok, :u:use_double, :u:teardown_ok)",
    )
    .unwrap();
    assert_eq!(scalar(&v), 84.0, "ok(fixture) unwraps to the fixture");
}

#[test]
fn setup_failure_skips_use_and_teardown() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    // teardown_err would surface if teardown ran; use_double would
    // hard-error on a missing fixture if use ran.
    let v = eval_value(
        &mut env,
        "err_message(bracket(:u:setup_err, :u:use_double, :u:teardown_err))",
    )
    .unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "no resource"),
        "setup's err is the result, untouched: {v:?}"
    );
}

#[test]
fn use_failure_stays_primary_and_teardown_still_runs() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let v = eval_value(
        &mut env,
        "err_message(bracket(:u:setup_plain, :u:use_err, :u:teardown_ok))",
    )
    .unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "use failed"), "{v:?}");
    // teardown ran: when use also fails, primary stays use's,
    // carried in the merged record's message field (full merge
    // coverage: bracket_error_tests).
    eval_value(&mut env, "def u:pluck(e) { e }").unwrap();
    let v = eval_value(
        &mut env,
        "p = or_else(:u:pluck, bracket(:u:setup_plain, :u:use_err, :u:teardown_err))\np.message",
    )
    .unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "use failed"), "{v:?}");
}

#[test]
fn teardown_failure_after_success_is_the_result() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let v = eval_value(
        &mut env,
        "err_message(bracket(:u:setup_plain, :u:use_double, :u:teardown_err))",
    )
    .unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "teardown leaked"),
        "a leaked resource is a real failure: {v:?}"
    );
}

#[test]
fn misuse_is_a_structured_error_naming_bracket() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let e = eval_value(&mut env, "bracket(:u:setup_plain, :u:use_double)").unwrap_err();
    assert!(e.contains("bracket"), "arity: {e}");
    let e = eval_value(&mut env, "bracket(1, :u:use_double, :u:teardown_ok)").unwrap_err();
    assert!(
        e.contains("bracket") && e.contains("reference"),
        "non-reference: {e}"
    );
    let e = eval_value(&mut env, "bracket(:mean, :u:use_double, :u:teardown_ok)").unwrap_err();
    assert!(
        e.contains("bracket") && e.contains("u:"),
        "builtin refs are rejected -- lifecycle hooks are user code: {e}"
    );
}
