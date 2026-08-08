//! record_keys(record) -> string-list of keys in deterministic
//! (sorted) order, and duplicate-key rejection in parse_json --
//! the demo-ml-utils step-004 unblock (safetensors tensor-name
//! discovery + duplicate-name validation).

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
fn keys_are_returned_sorted() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(
            &mut env,
            "equal([\"a\", \"b\", \"c\"], record_keys({b: 2, a: 1, c: 3}))"
        ),
        1.0
    );
}

#[test]
fn keys_of_a_parsed_record() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "r = unwrap(parse_json(\"{\\\"z\\\": 1, \\\"a\\\": 2}\"))",
    )
    .unwrap();
    assert_eq!(
        scalar(&mut env, "equal([\"a\", \"z\"], record_keys(r))"),
        1.0
    );
}

#[test]
fn empty_record_has_no_keys() {
    let mut env = Environment::new();
    eval_value(&mut env, "e = unwrap(parse_json(\"{}\"))").unwrap();
    let v = eval_value(&mut env, "record_keys(e)").unwrap();
    assert!(
        matches!(&v, Value::StrList { items } if items.is_empty()),
        "expected empty string-list, got {v:?}"
    );
}

#[test]
fn non_record_is_a_hard_error() {
    let mut env = Environment::new();
    assert!(eval_value(&mut env, "record_keys([1, 2, 3])").is_err());
    assert!(eval_value(&mut env, "record_keys(5)").is_err());
    assert!(eval_value(&mut env, "record_keys(\"hi\")").is_err());
}

#[test]
fn parse_json_rejects_duplicate_keys() {
    let mut env = Environment::new();
    // duplicate member -> err Result (evidence for duplicate-name validation)
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_json(\"{\\\"a\\\": 1, \\\"a\\\": 2}\"))"
        ),
        0.0
    );
    // distinct keys still parse
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_json(\"{\\\"a\\\": 1, \\\"b\\\": 2}\"))"
        ),
        1.0
    );
}
