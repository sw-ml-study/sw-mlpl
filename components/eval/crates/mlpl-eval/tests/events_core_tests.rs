//! test_event_sink / emit_test_event -- the typed test-event
//! API (mlplunit's native-test-events contract): envelope
//! validation is loud, delivery goes to the registered user-fn
//! sink, and unknown ADDITIVE fields pass through untouched.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

const EVENT: &str = "{version: 1, kind: \"test_end\", suite: \"capabilities\", \
                     name: \"event contract\", status: \"passed\", \
                     source: \"tests/x.mlpl\", line: 8}";

#[test]
fn sink_registers_and_receives_the_event() {
    let mut env = Environment::new();
    // The sink inspects the record; a mismatch would err.
    eval_value(
        &mut env,
        "def u:sink(e) { if eq(equal(e.kind, \"test_end\"), 1) { ok(1) } else { err(\"wrong kind\") } }",
    )
    .unwrap();
    let v = eval_value(&mut env, "is_ok(test_event_sink(:u:sink))").unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0));
    let v = eval_value(&mut env, &format!("is_ok(emit_test_event({EVENT}))")).unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0), "{v:?}");
}

#[test]
fn sink_failure_becomes_emits_failure() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:sink(e) { err(e.name) }").unwrap();
    eval_value(&mut env, "test_event_sink(:u:sink)").unwrap();
    let v = eval_value(&mut env, &format!("err_message(emit_test_event({EVENT}))")).unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "event contract"),
        "the sink SAW the record and its failure surfaced: {v:?}"
    );
}

#[test]
fn no_sink_means_validate_only() {
    let mut env = Environment::new();
    let v = eval_value(&mut env, &format!("is_ok(emit_test_event({EVENT}))")).unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0));
}

#[test]
fn envelope_validation_is_loud() {
    let mut env = Environment::new();
    // Unknown kind, listing the known ones.
    let e = eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"exploded\", suite: \"s\", name: \"n\"})",
    )
    .unwrap_err();
    assert!(e.contains("exploded") && e.contains("test_end"), "{e}");
    // Incompatible version.
    let e = eval_value(
        &mut env,
        "emit_test_event({version: 2, kind: \"suite_start\", suite: \"s\", name: \"n\"})",
    )
    .unwrap_err();
    assert!(e.contains("version"), "{e}");
    // *_end without a valid status.
    let e = eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"test_end\", suite: \"s\", name: \"n\", status: \"exploded\"})",
    )
    .unwrap_err();
    assert!(
        e.contains("status") && e.contains("expected_failure"),
        "{e}"
    );
    // Missing name.
    let e = eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"output\", suite: \"s\"})",
    )
    .unwrap_err();
    assert!(e.contains("name"), "{e}");
    // Not a record at all.
    let e = eval_value(&mut env, "emit_test_event(42)").unwrap_err();
    assert!(e.contains("record"), "{e}");
}

#[test]
fn additive_fields_pass_through_and_sink_reads_them() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:sink(e) { if eq(e.retries, 3) { ok(1) } else { err(\"lost the additive field\") } }",
    )
    .unwrap();
    eval_value(&mut env, "test_event_sink(:u:sink)").unwrap();
    let v = eval_value(
        &mut env,
        "is_ok(emit_test_event({version: 1, kind: \"test_start\", suite: \"s\", name: \"n\", retries: 3}))",
    )
    .unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0));
}

#[test]
fn sink_misuse_is_structured() {
    let mut env = Environment::new();
    let e = eval_value(&mut env, "test_event_sink(:mean)").unwrap_err();
    assert!(e.contains("u:"), "builtin refs rejected: {e}");
    let e = eval_value(&mut env, "test_event_sink(1)").unwrap_err();
    assert!(e.contains("reference"), "{e}");
}
