//! has_field / record_get -- exception-free record schema access.

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

const R: &str = "r = {name: \"gpt\", layers: 12, cfg: {heads: 8}}";

#[test]
fn has_field_tests_presence() {
    let mut env = Environment::new();
    eval_value(&mut env, R).unwrap();
    assert_eq!(scalar(&mut env, "has_field(r, \"name\")"), 1.0);
    assert_eq!(scalar(&mut env, "has_field(r, \"layers\")"), 1.0);
    assert_eq!(scalar(&mut env, "has_field(r, \"missing\")"), 0.0);
    // nested record is a value like any other
    assert_eq!(scalar(&mut env, "has_field(r.cfg, \"heads\")"), 1.0);
    assert_eq!(scalar(&mut env, "has_field(r.cfg, \"nope\")"), 0.0);
}

#[test]
fn record_get_returns_ok_or_err() {
    let mut env = Environment::new();
    eval_value(&mut env, R).unwrap();
    // present -> ok(value); unwrap it
    assert_eq!(scalar(&mut env, "unwrap(record_get(r, \"layers\"))"), 12.0);
    let v = eval_value(&mut env, "unwrap(record_get(r, \"name\"))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "gpt"));
    // absent -> err (is_err = 1)
    assert_eq!(scalar(&mut env, "is_err(record_get(r, \"missing\"))"), 1.0);
    // the err payload is structured and names the field
    eval_value(&mut env, "def u:pluck(e) { e }").unwrap();
    let e = eval_value(
        &mut env,
        "or_else(:u:pluck, record_get(r, \"missing\")).field",
    )
    .unwrap();
    assert!(
        matches!(&v, Value::Str(_)) && matches!(&e, Value::Str(s) if s == "missing"),
        "{e:?}"
    );
}

#[test]
fn exception_free_validation_pattern() {
    // The downstream use: validate a schema without try/catch.
    let mut env = Environment::new();
    eval_value(&mut env, R).unwrap();
    eval_value(
        &mut env,
        "def u:valid(cfg) { has_field(cfg, \"name\") * has_field(cfg, \"layers\") }",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "u:valid(r)"), 1.0);
    assert_eq!(scalar(&mut env, "u:valid({name: \"x\"})"), 0.0);
}

#[test]
fn type_errors_are_loud() {
    let mut env = Environment::new();
    eval_value(&mut env, R).unwrap();
    assert!(eval_value(&mut env, "has_field(42, \"x\")").is_err());
    assert!(eval_value(&mut env, "has_field(r, 5)").is_err());
    assert!(eval_value(&mut env, "record_get([1, 2], \"x\")").is_err());
}
