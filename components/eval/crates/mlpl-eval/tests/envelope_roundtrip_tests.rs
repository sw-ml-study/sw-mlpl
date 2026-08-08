//! Tagged-envelope round trip: parse_json reconstructs the
//! reserved $mlpl envelopes to_json({tagged: 1}) emits --
//! UNCONDITIONALLY (the reserved key is never application data) --
//! so a rank->=2 array or a Result survives a JSON round trip.

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

/// Round-trip helper: to_json tagged, then parse_json (unwraps).
fn rt(src: &str) -> String {
    format!("unwrap(parse_json(unwrap(to_json({src}, {{tagged: 1}}))))")
}

#[test]
fn rank2_array_round_trips() {
    let mut env = Environment::new();
    let m = "reshape(range(4), [2, 2])";
    assert_eq!(scalar(&mut env, &format!("equal({m}, {})", rt(m))), 1.0);
    // and it comes back a genuine rank-2 array
    assert_eq!(scalar(&mut env, &format!("rank({})", rt(m))), 2.0);
}

#[test]
fn results_round_trip_unconditionally() {
    let mut env = Environment::new();
    // NB: no {results: 1} needed -- the $mlpl envelope is unambiguous
    assert_eq!(
        scalar(&mut env, &format!("equal(ok(5), {})", rt("ok(5)"))),
        1.0
    );
    assert_eq!(
        scalar(
            &mut env,
            &format!("equal(err(\"boom\"), {})", rt("err(\"boom\")"))
        ),
        1.0
    );
    assert_eq!(
        kind(&mut env, &format!("type_of({})", rt("ok(5)"))),
        "result"
    );
}

#[test]
fn nested_envelopes_round_trip() {
    let mut env = Environment::new();
    eval_value(&mut env, "v = {m: reshape(range(4), [2, 2]), r: ok(7)}").unwrap();
    assert_eq!(scalar(&mut env, &format!("equal(v, {})", rt("v"))), 1.0);
}

#[test]
fn record_with_dollar_mlpl_key_round_trips_via_escape() {
    let mut env = Environment::new();
    // a genuine data record whose key is literally "$mlpl"
    eval_value(&mut env, "v = unwrap(parse_json(\"{\\\"$mlpl\\\": 1}\"))").unwrap();
    assert_eq!(scalar(&mut env, &format!("equal(v, {})", rt("v"))), 1.0);
    assert_eq!(kind(&mut env, &format!("type_of({})", rt("v"))), "record");
}

#[test]
fn plain_json_is_unaffected() {
    let mut env = Environment::new();
    // an ordinary object without $mlpl decodes to a plain record
    assert_eq!(
        kind(
            &mut env,
            "type_of(unwrap(parse_json(\"{\\\"a\\\": 1, \\\"b\\\": 2}\")))"
        ),
        "record"
    );
}
