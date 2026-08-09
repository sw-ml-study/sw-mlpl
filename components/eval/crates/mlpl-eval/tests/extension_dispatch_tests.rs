//! Native-extension invocation through the registry: the static
//! `hello` provider registers, then colon-qualified calls
//! (`hello:answer()`) dispatch through the boundary. Domain
//! failures are err Results; panics are contained as hard errors.

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
fn registered_extension_call_returns_its_value() {
    mlpl_ext_hello_static::register();
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "hello:answer()"), 42.0);
}

#[test]
fn failing_extension_call_is_an_err_result() {
    mlpl_ext_hello_static::register();
    let mut env = Environment::new();
    // hello:fail returns a domain error -> an err Result, not a hard error
    assert_eq!(scalar(&mut env, "is_ok(hello:fail())"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(ok(hello:answer()))"), 1.0);
}

#[test]
fn panicking_extension_is_contained_as_a_hard_error() {
    mlpl_ext_hello_static::register();
    let mut env = Environment::new();
    // hello:boom panics; the host must contain it as an EvalError,
    // never an unwind through the interpreter.
    let e = eval_value(&mut env, "hello:boom()").unwrap_err();
    assert!(
        e.contains("extension error") && e.contains("hello:boom"),
        "expected contained extension error, got: {e}"
    );
}

#[test]
fn unregistered_colon_name_still_routes_to_user_fns() {
    mlpl_ext_hello_static::register();
    let mut env = Environment::new();
    // a colon name that is NOT a registered extension must still
    // fall through to user-fn routing (here: an undefined user fn).
    assert!(eval_value(&mut env, "nosuch:thing()").is_err());
}
