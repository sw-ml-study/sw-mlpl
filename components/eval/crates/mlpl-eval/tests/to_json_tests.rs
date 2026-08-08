//! to_json(value) -- deterministic JSON encoding; the encode
//! half of the parse_json <-> to_json round trip. Result-based:
//! ok(json_string) on success, err(message) for a non-data kind
//! or a non-finite number (JSON has no NaN / infinity).

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

/// Pull the string out of a `to_json` OK result.
fn json(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => match *payload {
            Value::Str(s) => s,
            other => panic!("expected ok(string) from {src}, got ok({other:?})"),
        },
        other => panic!("expected ok(...) result from {src}, got {other:?}"),
    }
}

/// Pull the message out of a `to_json` ERR result.
fn err_msg(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: false, payload } => match *payload {
            Value::Str(s) => s,
            other => panic!("expected err(string) from {src}, got err({other:?})"),
        },
        other => panic!("expected err(...) result from {src}, got {other:?}"),
    }
}

#[test]
fn scalars_arrays_strings() {
    let mut env = Environment::new();
    assert_eq!(json(&mut env, "to_json(42)"), "42");
    assert_eq!(json(&mut env, "to_json(3.5)"), "3.5");
    assert_eq!(json(&mut env, "to_json([1, 2, 3])"), "[1,2,3]");
    assert_eq!(json(&mut env, "to_json([\"a\", \"b\"])"), r#"["a","b"]"#);
}

#[test]
fn records_have_sorted_keys() {
    let mut env = Environment::new();
    assert_eq!(json(&mut env, "to_json({b: 2, a: 1})"), r#"{"a":1,"b":2}"#);
    assert_eq!(json(&mut env, "to_json({x: {y: 7}})"), r#"{"x":{"y":7}}"#);
}

#[test]
fn strings_escape_and_unicode_is_exact() {
    let mut env = Environment::new();
    assert_eq!(json(&mut env, "to_json(\"a\\\"b\")"), r#""a\"b""#);
    assert_eq!(json(&mut env, "to_json(\"caf\u{e9}\")"), "\"caf\u{e9}\"");
}

#[test]
fn results_encode_with_ok_and_value() {
    let mut env = Environment::new();
    assert_eq!(json(&mut env, "to_json(ok(1))"), r#"{"ok":true,"value":1}"#);
    assert_eq!(
        json(&mut env, "to_json(err(\"bad\"))"),
        r#"{"ok":false,"error":"bad"}"#
    );
}

#[test]
fn round_trips_through_parse_json() {
    let mut env = Environment::new();
    for v in [
        "42",
        "3.5",
        "[1, 2, 3]",
        "[\"a\", \"b\"]",
        "{a: 1, b: \"x\"}",
    ] {
        eval_value(&mut env, &format!("v = {v}")).unwrap();
        // both codecs are Result-based now, so unwrap each.
        let back =
            eval_value(&mut env, "equal(unwrap(parse_json(unwrap(to_json(v)))), v)").unwrap();
        assert!(
            matches!(&back, Value::Array(a) if a.data() == [1.0]),
            "round trip failed for {v}: {back:?}"
        );
    }
}

#[test]
fn non_serializable_kinds_are_err_results() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:f(x) { x }").unwrap();
    // an err Result, NOT a hard error.
    assert_eq!(
        eval_value(&mut env, "is_ok(to_json(:u:f))")
            .unwrap()
            .to_string(),
        "0"
    );
    let msg = err_msg(&mut env, "to_json(:u:f)");
    assert!(msg.contains("user-fn-ref"), "message names the kind: {msg}");
    // a model is a non-data kind too (inlined so no cross-call binding).
    assert_eq!(
        eval_value(&mut env, "is_ok(to_json(chain(linear(2, 2, 0))))")
            .unwrap()
            .to_string(),
        "0"
    );
}

#[test]
fn non_finite_numbers_are_err_results() {
    let mut env = Environment::new();
    // 1/0 = +inf and sqrt(-1) = NaN have no JSON representation.
    let msg = err_msg(&mut env, "to_json(1 / 0)");
    assert!(msg.contains("finite"), "message explains non-finite: {msg}");
    assert_eq!(
        eval_value(&mut env, "is_ok(to_json(sqrt(0 - 1)))")
            .unwrap()
            .to_string(),
        "0"
    );
    // a non-finite cell inside an array is caught too.
    assert_eq!(
        eval_value(&mut env, "is_ok(to_json([1, 1 / 0, 3]))")
            .unwrap()
            .to_string(),
        "0"
    );
    // finite values still succeed.
    assert_eq!(json(&mut env, "to_json([1, 2, 3])"), "[1,2,3]");
}
