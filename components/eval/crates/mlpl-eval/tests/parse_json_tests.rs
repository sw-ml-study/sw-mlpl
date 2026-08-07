//! parse_json(s) -- JSON text to typed MLPL values (mlplunit's
//! structured-event-consumption gate): the inverse of the
//! test-event encoder.

use mlpl_eval::env_api::*;
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
fn their_fixture_shape_parses_to_a_typed_record() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "line = \"{\\\"version\\\":1,\\\"kind\\\":\\\"test_end\\\",\\\"name\\\":\\\"Unicode café report\\\",\\\"status\\\":\\\"passed\\\"}\"",
    )
    .unwrap();
    let v = eval_value(&mut env, "event = unwrap(parse_json(line))\nevent.kind").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "test_end"));
    let v = eval_value(&mut env, "event.name").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "Unicode café report"),
        "unicode exact: {v:?}"
    );
    assert_eq!(scalar(&mut env, "event.version"), 1.0);
}

#[test]
fn kinds_map_onto_mlpl_values() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "unwrap(parse_json(\"42.5\"))"), 42.5);
    assert_eq!(scalar(&mut env, "unwrap(parse_json(\"-3\"))"), -3.0);
    assert_eq!(scalar(&mut env, "unwrap(parse_json(\"true\"))"), 1.0);
    assert_eq!(scalar(&mut env, "unwrap(parse_json(\"false\"))"), 0.0);
    // null -> the empty vector (zilde: absence as data).
    assert_eq!(scalar(&mut env, "tally(unwrap(parse_json(\"null\")))"), 0.0);
    let v = eval_value(&mut env, "unwrap(parse_json(\"[1, 2, 3]\"))").unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data() == [1.0, 2.0, 3.0]));
    let v = eval_value(&mut env, "unwrap(parse_json(\"[\\\"a\\\", \\\"b\\\"]\"))").unwrap();
    assert!(matches!(&v, Value::StrList { items } if items == &["a", "b"]));
    // Nested objects recurse.
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(parse_json(\"{\\\"a\\\":{\\\"b\\\":7}}\")).a.b"
        ),
        7.0
    );
}

#[test]
fn escapes_and_unicode_are_exact() {
    let mut env = Environment::new();
    env.set_string(
        "j".into(),
        r#""say \"hi\" and \u2373 caf\u00e9""#.to_string(),
    );
    let v = eval_value(&mut env, "unwrap(parse_json(j))").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "say \"hi\" and \u{2373} caf\u{e9}"),
        "{v:?}"
    );
}

#[test]
fn event_encoder_round_trips_through_the_decoder() {
    let mut env = Environment::new();
    env.test_event_lines = Some(Vec::new());
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"test_end\", suite: \"s\", name: \"caf\u{e9} \u{2373}\", status: \"passed\", line: 12, tags: [\"a\", \"b\"]})",
    )
    .unwrap();
    let line = env.test_event_lines.as_ref().unwrap()[0].clone();
    env.set_string("line".into(), line);
    let v = eval_value(
        &mut env,
        "e = unwrap(parse_json(line))\nequal({k: e.kind, n: e.name, l: e.line, t: e.tags}, {k: \"test_end\", n: \"caf\u{e9} \u{2373}\", l: 12, t: [\"a\", \"b\"]})",
    )
    .unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data() == [1.0]), "{v:?}");
}

#[test]
fn malformed_json_is_an_err_with_position() {
    let mut env = Environment::new();
    for bad in ["{\\\"a\\\": }", "[1, ", "{\\\"a\\\" 1}", "nope"] {
        let v = eval_value(&mut env, &format!("err_message(parse_json(\"{bad}\"))")).unwrap();
        assert!(
            matches!(&v, Value::Str(s) if s.contains("parse_json")),
            "{bad}: {v:?}"
        );
    }
    // Mixed arrays are not representable.
    let v = eval_value(&mut env, "err_message(parse_json(\"[1, \\\"a\\\"]\"))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s.contains("mixed")), "{v:?}");
    // Wrong argument type is a hard error, not an err value.
    let e = eval_value(&mut env, "parse_json(42)").unwrap_err();
    assert!(e.contains("parse_json"), "{e}");
}
