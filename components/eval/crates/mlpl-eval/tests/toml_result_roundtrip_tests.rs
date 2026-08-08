//! to_toml encodes a Result FIELD as a `{ok, value|error}`
//! sub-table (rather than erroring), so Results round-trip
//! through TOML with parse_toml's `{results: 1}` option -- the
//! symmetric counterpart to the JSON codec.

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
fn to_toml_encodes_a_result_field() {
    let mut env = Environment::new();
    // no longer an err -- a Result field is representable
    assert_eq!(scalar(&mut env, "is_ok(to_toml({r: ok(5)}))"), 1.0);
    assert_eq!(scalar(&mut env, "is_ok(to_toml({r: err(\"bad\")}))"), 1.0);
}

#[test]
fn ok_field_round_trips_through_toml() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "equal({r: ok(5)}, unwrap(parse_toml(unwrap(to_toml({r: ok(5)})), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn err_field_round_trips_through_toml() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "equal({r: err(\"bad\")}, unwrap(parse_toml(unwrap(to_toml({r: err(\"bad\")})), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn result_with_record_payload_round_trips() {
    let mut env = Environment::new();
    eval_value(&mut env, "v = {r: ok({a: 1, b: 2})}").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal(v, unwrap(parse_toml(unwrap(to_toml(v)), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn result_at_root_still_errs_toml_needs_a_table() {
    let mut env = Environment::new();
    // a TOML document is a table; a bare Result root is not representable
    assert_eq!(scalar(&mut env, "is_ok(to_toml(ok(5)))"), 0.0);
}
