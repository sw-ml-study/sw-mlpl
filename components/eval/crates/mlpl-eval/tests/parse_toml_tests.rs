//! parse_toml(text) -- the decode half of the TOML codec. A TOML
//! document is a table, so the result is always a record wrapped
//! in a Result: ok(record) / err(message). Supports the config
//! subset: comments, key = value, [table] / dotted headers,
//! integer / float / boolean / basic-string / array values.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

fn text(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Str(s) => s,
        other => panic!("expected string from {src}, got {other:?}"),
    }
}

#[test]
fn scalars_strings_and_types() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"a = 1\\nb = 3.5\\ns = \\\"hi\\\"\"))",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"a\"))"), 1.0);
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"b\"))"), 3.5);
    assert_eq!(text(&mut env, "unwrap(record_get(r, \"s\"))"), "hi");
}

#[test]
fn booleans_become_one_and_zero() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"on = true\\noff = false\"))",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"on\"))"), 1.0);
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"off\"))"), 0.0);
}

#[test]
fn arrays_of_numbers_and_strings() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"nums = [1, 2, 3]\\ntags = [\\\"a\\\", \\\"b\\\"]\"))",
    )
    .unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal([1, 2, 3], unwrap(record_get(r, \"nums\")))"
        ),
        1.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "equal([\"a\", \"b\"], unwrap(record_get(r, \"tags\")))"
        ),
        1.0
    );
}

#[test]
fn comments_and_blank_lines_are_skipped() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"# header\\n\\na = 1  # trailing comment\\n\"))",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"a\"))"), 1.0);
}

#[test]
fn tables_and_dotted_headers_nest() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"a = 1\\n[sub]\\nb = 2\\n[sub.deep]\\nc = 3\"))",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"a\"))"), 1.0);
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(record_get(unwrap(record_get(r, \"sub\")), \"b\"))"
        ),
        2.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(record_get(unwrap(record_get(unwrap(record_get(r, \"sub\")), \"deep\")), \"c\"))"
        ),
        3.0
    );
}

#[test]
fn round_trips_with_to_toml() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "v = {a: 1, b: \"x\", nums: [1, 2, 3], sub: {c: 3.5}}",
    )
    .unwrap();
    assert_eq!(
        scalar(&mut env, "equal(v, unwrap(parse_toml(unwrap(to_toml(v)))))"),
        1.0
    );
}

#[test]
fn malformed_input_is_err() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"= 1\"))"), 0.0); // no key
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"a = \"))"), 0.0); // empty value
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"[unclosed\"))"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"a = @bad\"))"), 0.0);
}

#[test]
fn empty_document_is_an_empty_record() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"\"))"), 1.0);
    assert_eq!(
        scalar(&mut env, "is_ok(parse_toml(\"   \\n# just a comment\\n\"))"),
        1.0
    );
}
