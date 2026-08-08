//! Decode limits for parse_json / parse_toml: an optional options
//! record `{max_depth, max_bytes}` caps nesting depth (guarding
//! the recursive-descent decoder against stack overflow) and
//! input size. One-arg calls keep default limits. Bad input is an
//! err Result; a malformed options argument is a hard error.

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

#[test]
fn one_arg_calls_keep_default_limits() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(parse_json(\"{\\\"a\\\": 1}\"))"),
        1.0
    );
    assert_eq!(scalar(&mut env, "is_ok(parse_toml(\"a = 1\"))"), 1.0);
}

#[test]
fn json_depth_cap_rejects_too_deep() {
    let mut env = Environment::new();
    // three nested objects
    let deep = "parse_json(\"{\\\"a\\\": {\\\"a\\\": {\\\"a\\\": 1}}}\"";
    assert_eq!(
        scalar(&mut env, &format!("is_ok({deep}, {{max_depth: 2}}))")),
        0.0
    );
    assert_eq!(
        scalar(&mut env, &format!("is_ok({deep}, {{max_depth: 3}}))")),
        1.0
    );
}

#[test]
fn json_byte_cap_rejects_oversized() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(parse_json(\"[1, 2, 3]\", {max_bytes: 3}))"),
        0.0
    );
    assert_eq!(
        scalar(&mut env, "is_ok(parse_json(\"1\", {max_bytes: 10}))"),
        1.0
    );
}

#[test]
fn toml_byte_cap_rejects_oversized() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = 1\\nb = 2\", {max_bytes: 3}))"
        ),
        0.0
    );
}

#[test]
fn toml_depth_cap_applies_to_array_values() {
    let mut env = Environment::new();
    // a scalar value costs no depth
    assert_eq!(
        scalar(&mut env, "is_ok(parse_toml(\"a = 1\", {max_depth: 0}))"),
        1.0
    );
    // an array value opens one container -> needs depth >= 1
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = [1, 2]\", {max_depth: 0}))"
        ),
        0.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = [1, 2]\", {max_depth: 1}))"
        ),
        1.0
    );
}

#[test]
fn malformed_options_are_a_hard_error() {
    let mut env = Environment::new();
    // negative depth, non-integer, and a non-record options arg
    assert!(eval_value(&mut env, "parse_json(\"1\", {max_depth: 0 - 1})").is_err());
    assert!(eval_value(&mut env, "parse_json(\"1\", {max_bytes: 2.5})").is_err());
    assert!(eval_value(&mut env, "parse_json(\"1\", 5)").is_err());
    assert!(eval_value(&mut env, "parse_toml(\"a = 1\", 5)").is_err());
}
