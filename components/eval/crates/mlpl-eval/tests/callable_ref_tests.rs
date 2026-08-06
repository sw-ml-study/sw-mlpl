//! `:u:name` -- the quoted user-function form (callables design,
//! step callables-ref): one token, a distinct value kind, storable
//! in variables and record registries, honest under equal/repr.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

#[test]
fn user_ref_is_one_token_with_a_distinct_kind() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(x) { x * 2 }").unwrap();
    let v = eval_value(&mut env, ":u:double").unwrap();
    assert!(
        matches!(&v, Value::UserFnRef { name } if name == "u:double"),
        "{v:?}"
    );
    assert_eq!(mlpl_eval::value_kind(&v), "user-fn-ref");
    assert_eq!(format!("{v}"), ":u:double");
}

#[test]
fn refs_bind_to_variables_and_round_trip() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(x) { x * 2 }").unwrap();
    let v = eval_value(&mut env, "f = :u:double\nf").unwrap();
    assert!(
        matches!(&v, Value::UserFnRef { name } if name == "u:double"),
        "{v:?}"
    );
}

#[test]
fn record_registries_hold_and_return_refs() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:double(x) { x * 2 }\ndef u:halve(x) { x / 2 }",
    )
    .unwrap();
    let v = eval_value(
        &mut env,
        "suite = {double: :u:double, halve: :u:halve}\nsuite.halve",
    )
    .unwrap();
    assert!(
        matches!(&v, Value::UserFnRef { name } if name == "u:halve"),
        "{v:?}"
    );
}

#[test]
fn equal_and_repr_are_honest_about_refs() {
    let mut env = Environment::new();
    eval_value(&mut env, "def u:double(x) { x * 2 }").unwrap();
    let eq = eval_value(&mut env, "equal(:u:double, :u:double)").unwrap();
    assert!(matches!(&eq, Value::Array(a) if a.data() == [1.0]));
    let ne = eval_value(&mut env, "equal(:u:double, :double)").unwrap();
    assert!(
        matches!(&ne, Value::Array(a) if a.data() == [0.0]),
        "user ref != builtin ref of the same stem"
    );
    let Value::Str(r) = eval_value(&mut env, "repr(:u:double)").unwrap() else {
        panic!()
    };
    assert_eq!(r, ":u:double");
}

#[test]
fn plain_u_ref_stays_a_builtin_ref() {
    let mut env = Environment::new();
    let v = eval_value(&mut env, ":u").unwrap();
    assert!(
        matches!(&v, Value::BuiltinRef { name } if name == "u"),
        "{v:?}"
    );
}
