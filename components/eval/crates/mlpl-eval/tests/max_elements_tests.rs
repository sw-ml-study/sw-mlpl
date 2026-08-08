//! max_elements: a cumulative collection decode limit for
//! parse_json / parse_toml, beside max_depth and max_bytes. Each
//! object/record field, array cell, and string-list item consumes
//! one unit of the budget; exhausting it is an err Result.

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
fn json_object_field_count_is_capped() {
    let mut env = Environment::new();
    let doc = "parse_json(\"{\\\"a\\\":1,\\\"b\\\":2,\\\"c\\\":3}\"";
    assert_eq!(
        scalar(&mut env, &format!("is_ok({doc}, {{max_elements: 3}}))")),
        1.0
    );
    assert_eq!(
        scalar(&mut env, &format!("is_ok({doc}, {{max_elements: 2}}))")),
        0.0
    );
}

#[test]
fn json_array_cell_count_is_capped() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_json(\"[1,2,3,4]\", {max_elements: 4}))"
        ),
        1.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_json(\"[1,2,3,4]\", {max_elements: 3}))"
        ),
        0.0
    );
}

#[test]
fn nested_counts_are_cumulative() {
    let mut env = Environment::new();
    // 2 fields + 4 array cells = 6 elements
    let doc = "parse_json(\"{\\\"a\\\":[1,2],\\\"b\\\":[3,4]}\"";
    assert_eq!(
        scalar(&mut env, &format!("is_ok({doc}, {{max_elements: 6}}))")),
        1.0
    );
    assert_eq!(
        scalar(&mut env, &format!("is_ok({doc}, {{max_elements: 5}}))")),
        0.0
    );
}

#[test]
fn default_is_unbounded() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(parse_json(\"[1,2,3,4,5,6,7,8]\"))"),
        1.0
    );
}

#[test]
fn toml_counts_fields_and_array_cells() {
    let mut env = Environment::new();
    // three top-level fields
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = 1\\nb = 2\\nc = 3\", {max_elements: 3}))"
        ),
        1.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = 1\\nb = 2\\nc = 3\", {max_elements: 2}))"
        ),
        0.0
    );
    // one field + three array cells = 4
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = [1, 2, 3]\", {max_elements: 4}))"
        ),
        1.0
    );
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_toml(\"a = [1, 2, 3]\", {max_elements: 3}))"
        ),
        0.0
    );
}

#[test]
fn malformed_max_elements_is_a_hard_error() {
    let mut env = Environment::new();
    assert!(eval_value(&mut env, "parse_json(\"1\", {max_elements: 0 - 1})").is_err());
    assert!(eval_value(&mut env, "parse_json(\"1\", {max_elements: 2.5})").is_err());
}
