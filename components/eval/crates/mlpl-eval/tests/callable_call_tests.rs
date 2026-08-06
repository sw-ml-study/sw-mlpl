//! `call(f, args...)` -- uniform invocation over user and builtin
//! references (callables design step callables-call; mirrors
//! mlplunit's callable_function_case fixture).

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
fn the_mlplunit_fixture_shape_works() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(value) { value * 2 }").unwrap();
    let v = eval_value(&mut env, "function = :u:double\ncall(function, 21)").unwrap();
    assert_eq!(scalar(&v), 42.0);
}

#[test]
fn direct_refs_registries_and_builtins_all_call() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(x) { x * 2 }").unwrap();
    assert_eq!(
        scalar(&eval_value(&mut env, "call(:u:double, 21)").unwrap()),
        42.0
    );
    assert_eq!(
        scalar(&eval_value(&mut env, "suite = {d: :u:double}\ncall(suite.d, 4)").unwrap()),
        8.0
    );
    assert_eq!(
        scalar(&eval_value(&mut env, "call(:mean, [2, 4])").unwrap()),
        3.0,
        "builtin references invoke through the same call"
    );
}

#[test]
fn errors_identify_the_referent_not_call() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(x) { x * 2 }").unwrap();
    let e = eval_value(&mut env, "call(:u:double)").unwrap_err();
    assert!(
        e.contains("u:double"),
        "arity error names the referent: {e}"
    );
    let e = eval_value(&mut env, "call(:u:gone, 1)").unwrap_err();
    assert!(e.contains("u:gone"), "dangling reference names it: {e}");
    let e = eval_value(&mut env, "call(5, 1)").unwrap_err();
    assert!(e.contains("function reference"), "tutoring error: {e}");
}

#[test]
fn result_semantics_flow_through_call() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:fails(x) { err({kind: \"nope\", message: \"boom\"}) }",
    )
    .unwrap();
    let v = eval_value(&mut env, "unwrap_or(call(:u:fails, 1), 9)").unwrap();
    assert_eq!(
        scalar(&v),
        9.0,
        "Err from the callee is a VALUE, not a crash"
    );
}
