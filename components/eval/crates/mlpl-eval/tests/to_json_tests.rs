//! to_json(value) -- deterministic JSON encoding; the encode
//! half of the parse_json <-> to_json round trip.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn json(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Str(s) => s,
        other => panic!("expected string from {src}, got {other:?}"),
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
    // insertion order b, a -> output sorted a, b
    assert_eq!(json(&mut env, "to_json({b: 2, a: 1})"), r#"{"a":1,"b":2}"#);
    // nested record
    assert_eq!(json(&mut env, "to_json({x: {y: 7}})"), r#"{"x":{"y":7}}"#);
}

#[test]
fn strings_escape_and_unicode_is_exact() {
    let mut env = Environment::new();
    assert_eq!(json(&mut env, "to_json(\"a\\\"b\")"), r#""a\"b""#);
    // a real glyph passes through untouched
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
        let back = eval_value(&mut env, "equal(unwrap(parse_json(to_json(v))), v)").unwrap();
        assert!(
            matches!(&back, Value::Array(a) if a.data() == [1.0]),
            "round trip failed for {v}: {back:?}"
        );
    }
}

#[test]
fn non_serializable_kinds_error() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:f(x) { x }").unwrap();
    assert!(eval_value(&mut env, "to_json(:u:f)").is_err());
    eval_value(&mut env, "m = chain(linear(2, 2, 0))").unwrap();
    assert!(eval_value(&mut env, "to_json(m)").is_err());
}
