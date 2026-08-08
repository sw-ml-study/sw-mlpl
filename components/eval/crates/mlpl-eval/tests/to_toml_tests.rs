//! to_toml(record) -- deterministic TOML encoding, the encode
//! half of the TOML codec. Result-based like to_json: ok(text) /
//! err(message). The root must be a record (a TOML document is a
//! table); fields emit sorted, scalars/strings/arrays first, then
//! nested records as [section] tables.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn toml(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => match *payload {
            Value::Str(s) => s,
            other => panic!("expected ok(string) from {src}, got ok({other:?})"),
        },
        other => panic!("expected ok(...) from {src}, got {other:?}"),
    }
}

fn is_ok(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

#[test]
fn scalars_and_strings() {
    let mut env = Environment::new();
    assert_eq!(toml(&mut env, "to_toml({a: 1, b: 2})"), "a = 1\nb = 2\n");
    assert_eq!(toml(&mut env, "to_toml({x: 3.5})"), "x = 3.5\n");
    assert_eq!(toml(&mut env, "to_toml({s: \"hi\"})"), "s = \"hi\"\n");
}

#[test]
fn keys_are_sorted() {
    let mut env = Environment::new();
    // insertion order b, a -> sorted a, b
    assert_eq!(toml(&mut env, "to_toml({b: 2, a: 1})"), "a = 1\nb = 2\n");
}

#[test]
fn arrays_of_numbers_and_strings() {
    let mut env = Environment::new();
    assert_eq!(
        toml(&mut env, "to_toml({nums: [1, 2, 3]})"),
        "nums = [1, 2, 3]\n"
    );
    assert_eq!(
        toml(&mut env, "to_toml({tags: [\"a\", \"b\"]})"),
        "tags = [\"a\", \"b\"]\n"
    );
}

#[test]
fn nested_records_become_sections() {
    let mut env = Environment::new();
    // scalars first, then the [sub] table, separated by a blank line
    assert_eq!(
        toml(&mut env, "to_toml({a: 1, sub: {b: 2}})"),
        "a = 1\n\n[sub]\nb = 2\n"
    );
    // dotted path for deeper nesting, no leading blank line
    assert_eq!(
        toml(&mut env, "to_toml({sub: {deep: {c: 3}}})"),
        "[sub]\n\n[sub.deep]\nc = 3\n"
    );
}

#[test]
fn string_escapes_are_applied() {
    let mut env = Environment::new();
    assert_eq!(
        toml(&mut env, "to_toml({s: \"a\\\"b\"})"),
        "s = \"a\\\"b\"\n"
    );
}

#[test]
fn non_record_root_is_err() {
    let mut env = Environment::new();
    assert_eq!(is_ok(&mut env, "is_ok(to_toml(42))"), 0.0);
    assert_eq!(is_ok(&mut env, "is_ok(to_toml([1, 2, 3]))"), 0.0);
    assert_eq!(is_ok(&mut env, "is_ok(to_toml(\"hi\"))"), 0.0);
}

#[test]
fn non_finite_and_non_representable_fields_are_err() {
    let mut env = Environment::new();
    assert_eq!(is_ok(&mut env, "is_ok(to_toml({x: 1 / 0}))"), 0.0);
    // a Result field has no TOML representation
    assert_eq!(is_ok(&mut env, "is_ok(to_toml({r: ok(1)}))"), 0.0);
    // rank>=2 array field is not a TOML value here
    assert_eq!(
        is_ok(&mut env, "is_ok(to_toml({m: [[1, 2], [3, 4]]}))"),
        0.0
    );
}
