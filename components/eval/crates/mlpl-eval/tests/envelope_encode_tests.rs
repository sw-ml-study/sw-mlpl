//! to_json(v, {tagged: 1}) -- the reserved $mlpl tagged envelope.
//! A rank->=2 array becomes a {shape, data} envelope and a Result
//! a {variant, value|error} envelope, so values plain JSON cannot
//! represent losslessly round-trip. Inner keys sort deterministically.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn json(env: &mut Environment, src: &str) -> String {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Result { ok: true, payload } => match *payload {
            Value::Str(s) => s,
            other => panic!("expected ok(string), got ok({other:?})"),
        },
        other => panic!("expected ok(...) from {src}, got {other:?}"),
    }
}

#[test]
fn rank2_array_becomes_a_shape_data_envelope() {
    let mut env = Environment::new();
    assert_eq!(
        json(&mut env, "to_json(reshape(range(4), [2, 2]), {tagged: 1})"),
        r#"{"$mlpl":{"data":[0,1,2,3],"shape":[2,2],"type":"array","v":1}}"#
    );
}

#[test]
fn ok_and_err_become_result_envelopes() {
    let mut env = Environment::new();
    assert_eq!(
        json(&mut env, "to_json(ok(5), {tagged: 1})"),
        r#"{"$mlpl":{"type":"result","v":1,"value":5,"variant":"ok"}}"#
    );
    assert_eq!(
        json(&mut env, "to_json(err(\"boom\"), {tagged: 1})"),
        r#"{"$mlpl":{"error":"boom","type":"result","v":1,"variant":"err"}}"#
    );
}

#[test]
fn plain_values_are_unchanged_in_tagged_mode() {
    let mut env = Environment::new();
    assert_eq!(
        json(&mut env, "to_json({b: 2, a: 1}, {tagged: 1})"),
        r#"{"a":1,"b":2}"#
    );
    assert_eq!(json(&mut env, "to_json([1, 2, 3], {tagged: 1})"), "[1,2,3]");
    assert_eq!(json(&mut env, "to_json(42, {tagged: 1})"), "42");
}

#[test]
fn envelopes_nest_inside_records() {
    let mut env = Environment::new();
    assert_eq!(
        json(
            &mut env,
            "to_json({m: reshape(range(4), [2, 2])}, {tagged: 1})"
        ),
        r#"{"m":{"$mlpl":{"data":[0,1,2,3],"shape":[2,2],"type":"array","v":1}}}"#
    );
}

#[test]
fn default_mode_is_unchanged_no_envelope() {
    let mut env = Environment::new();
    // without {tagged: 1}, ok(5) still uses the compact form
    assert_eq!(json(&mut env, "to_json(ok(5))"), r#"{"ok":true,"value":5}"#);
}
