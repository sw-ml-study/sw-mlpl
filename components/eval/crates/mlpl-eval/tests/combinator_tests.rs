//! map_ok / and_then / or_else -- the error monad's composition
//! over function references (docs/monads.md rider).

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(v: &Value) -> f64 {
    match v {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn map_ok_applies_inside_ok_and_skips_err() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:inc(x) { x + 1 }").unwrap();
    let v = eval_value(&mut env, "unwrap(map_ok(:u:inc, ok(41)))").unwrap();
    assert_eq!(scalar(&v), 42.0);
    let v = eval_value(&mut env, "err_message(map_ok(:u:inc, err(\"boom\")))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "boom"), "{v:?}");
}

#[test]
fn and_then_chains_the_railway() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:pos(x) { if gt(x, 0) { ok(x) } else { err(\"not positive\") } }\n\
         def u:halve(x) { ok(x / 2) }",
    )
    .unwrap();
    let v = eval_value(
        &mut env,
        "unwrap(and_then(:u:halve, and_then(:u:pos, ok(10))))",
    )
    .unwrap();
    assert_eq!(scalar(&v), 5.0);
    let v = eval_value(
        &mut env,
        "err_message(and_then(:u:halve, and_then(:u:pos, ok(0 - 4))))",
    )
    .unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "not positive"), "{v:?}");
}

#[test]
fn or_else_recovers_only_errors() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:fallback(e) { ok(9) }").unwrap();
    let v = eval_value(&mut env, "unwrap(or_else(:u:fallback, err(\"x\")))").unwrap();
    assert_eq!(scalar(&v), 9.0);
    let v = eval_value(&mut env, "unwrap(or_else(:u:fallback, ok(1)))").unwrap();
    assert_eq!(scalar(&v), 1.0, "ok passes through untouched");
}

#[test]
fn builtin_refs_work_on_array_payloads_and_errors_tutor() {
    let mut env = Environment::new();
    let v = eval_value(&mut env, "unwrap(map_ok(:sqrt, ok([4, 9])))").unwrap();
    assert!(
        matches!(&v, Value::Array(a) if a.data() == [2.0, 3.0]),
        "{v:?}"
    );
    let e = eval_value(&mut env, "map_ok(:u:inc, 5)").unwrap_err();
    assert!(e.contains("must be a Result"), "{e}");
    let e = eval_value(&mut env, "map_ok(5, ok(1))").unwrap_err();
    assert!(e.contains("function reference"), "{e}");
}
