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
    // Under-application is no longer an error: it forms a
    // PARTIAL (docs/combinators-design.md). call(:u:double)
    // with nothing supplied returns the callable, 0 of 1 bound.
    let v = eval_value(&mut env, "call(:u:double)").unwrap();
    assert!(
        matches!(&v, Value::Partial { name, bound, .. } if name == "u:double" && bound.is_empty()),
        "{v:?}"
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

#[test]
fn references_pass_through_udf_arguments() {
    // The strengthened mlplunit fixture shape: invoke a reference
    // that arrived as a user-function ARGUMENT, zero-arg and
    // one-arg, including through a record registry field.
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:double(x) { x * 2 }\n\
         def u:pass_test() { ok({test: \"pass\"}) }\n\
         def u:invoke_test(test) { call(test) }\n\
         def u:invoke_case(test, input) { call(test, input) }",
    )
    .unwrap();
    let v = eval_value(
        &mut env,
        "registry = {d: :u:double, p: :u:pass_test}\nu:invoke_case(registry.d, 21)",
    )
    .unwrap();
    assert_eq!(scalar(&v), 42.0);
    let v = eval_value(&mut env, "is_ok(u:invoke_test(registry.p))").unwrap();
    assert_eq!(scalar(&v), 1.0);
}

#[test]
fn reference_arguments_do_not_leak_out_of_the_frame() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:double(x) { x * 2 }\n\
         def u:invoke_case(test, input) { call(test, input) }\n\
         test = :u:double",
    )
    .unwrap();
    eval_value(
        &mut env,
        "def u:halve(x) { x / 2 }\nu:invoke_case(:u:halve, 10)",
    )
    .unwrap();
    // The caller's own `test` binding must survive the callee's
    // shadowing parameter of the same name.
    let v = eval_value(&mut env, "call(test, 3)").unwrap();
    assert_eq!(scalar(&v), 6.0, "caller's reference restored");
}
