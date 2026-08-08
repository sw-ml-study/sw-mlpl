//! Opt-in semantic Result reconstruction: parse_json /
//! parse_toml with `{results: 1}` rebuild the Result-shaped
//! records that to_json / to_toml emit ({ok, value} / {ok, error})
//! back into ok(...) / err(...) values, so Results survive a text
//! round trip. Off by default (the shape is an ordinary record).

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

fn kind(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Str(s) => s,
        other => panic!("expected string from {src}, got {other:?}"),
    }
}

#[test]
fn ok_value_round_trips_with_results_flag() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "equal(ok(5), unwrap(parse_json(unwrap(to_json(ok(5))), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn err_value_round_trips_with_results_flag() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "equal(err(\"bad\"), unwrap(parse_json(unwrap(to_json(err(\"bad\"))), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn nested_results_are_rebuilt() {
    let mut env = Environment::new();
    eval_value(&mut env, "v = ok({a: ok(1), b: err(\"no\")})").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal(v, unwrap(parse_json(unwrap(to_json(v)), {results: 1})))"
        ),
        1.0
    );
}

#[test]
fn reconstruction_is_opt_in_off_by_default() {
    let mut env = Environment::new();
    // without the flag, the encoded shape stays a plain record
    assert_eq!(
        kind(
            &mut env,
            "type_of(unwrap(parse_json(unwrap(to_json(ok(5))))))"
        ),
        "record"
    );
    // with the flag, it becomes a result
    assert_eq!(
        kind(
            &mut env,
            "type_of(unwrap(parse_json(unwrap(to_json(ok(5))), {results: 1})))"
        ),
        "result"
    );
}

#[test]
fn non_result_records_are_untouched_under_the_flag() {
    let mut env = Environment::new();
    // a plain data record with unrelated keys stays a record
    assert_eq!(
        kind(
            &mut env,
            "type_of(unwrap(parse_json(\"{\\\"a\\\": 1, \\\"b\\\": 2}\", {results: 1})))"
        ),
        "record"
    );
    // a nested record field survives (still a record with its value)
    eval_value(
        &mut env,
        "n = unwrap(parse_json(\"{\\\"a\\\": {\\\"x\\\": 7}}\", {results: 1}))",
    )
    .unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(record_get(unwrap(record_get(n, \"a\")), \"x\"))"
        ),
        7.0
    );
}

#[test]
fn toml_results_reconstruct_too() {
    let mut env = Environment::new();
    // a TOML sub-table shaped like an ok(...) rebuilds under the flag
    eval_value(
        &mut env,
        "r = unwrap(parse_toml(\"[res]\\nok = true\\nvalue = 9\", {results: 1}))",
    )
    .unwrap();
    assert_eq!(
        kind(&mut env, "type_of(unwrap(record_get(r, \"res\")))"),
        "result"
    );
    assert_eq!(
        scalar(&mut env, "unwrap(unwrap(record_get(r, \"res\")))"),
        9.0
    );
}
