//! Record destructuring (microgpt-mlpl request #5): `{a, b} = expr` binds
//! each named field to a variable of the same name, `{a, b: x} = expr`
//! binds field `b` to `x`, and `{a, b} = expr?` unwraps a Result first.
//! Extra fields are ignored; a missing field errors naming the field and
//! the fields present, and binds nothing. Inside a `u:` body the bindings
//! are frame-scoped like any assignment.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, EvalError> {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env)
}

fn num(env: &mut Environment, src: &str) -> f64 {
    match run(env, src) {
        Ok(Value::Array(a)) => a.data()[0],
        other => panic!("{src}: {other:?}"),
    }
}

#[test]
fn binds_fields_by_name_and_ignores_extras() {
    let mut env = Environment::new();
    run(&mut env, "{a, b} = {a: 1, b: 2, c: 3}").unwrap();
    assert_eq!(num(&mut env, "a * 10 + b"), 12.0);
    assert!(run(&mut env, "c").is_err(), "extra field must not bind");
}

#[test]
fn rename_binds_field_to_other_name() {
    let mut env = Environment::new();
    run(
        &mut env,
        "r = {loss: 0.5, grad: [1, 2]}\n{loss: l, grad} = r",
    )
    .unwrap();
    assert_eq!(num(&mut env, "l"), 0.5);
    assert_eq!(num(&mut env, "reduce_add(grad)"), 3.0);
    assert!(run(&mut env, "loss").is_err(), "renamed field binds only x");
}

#[test]
fn works_with_question_mark_and_strings() {
    let mut env = Environment::new();
    run(&mut env, "{name, n} = ok({name: \"tok\", n: 7})?").unwrap();
    assert_eq!(num(&mut env, "n"), 7.0);
    match run(&mut env, "name") {
        Ok(Value::Str(s)) => assert_eq!(s, "tok"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn missing_field_errors_and_binds_nothing() {
    let mut env = Environment::new();
    let err = run(&mut env, "{a, zz} = {a: 1, b: 2}").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("zz") && msg.contains('b'), "{msg}");
    assert!(
        run(&mut env, "a").is_err(),
        "atomic: nothing bound on error"
    );
    let err = run(&mut env, "{a} = 5").unwrap_err().to_string();
    assert!(err.contains("record"), "{err}");
}

#[test]
fn frame_scoped_inside_user_functions() {
    let mut env = Environment::new();
    let src = "def u:area(r) {\n  {w, h} = r\n  w * h\n}\nw = 100\nu:area({w: 3, h: 4})";
    assert_eq!(num(&mut env, src), 12.0);
    assert_eq!(num(&mut env, "w"), 100.0, "global w untouched");
    assert!(run(&mut env, "h").is_err(), "h must not leak");
}

#[test]
fn inside_grad_is_a_named_error() {
    let mut env = Environment::new();
    let src =
        "w = param[2]\ndef u:f(r) {\n  {x} = r\n  reduce_add(w * x)\n}\ngrad(u:f({x: [1, 2]}), w)";
    let msg = run(&mut env, src).unwrap_err().to_string();
    assert!(msg.contains("destructuring"), "{msg}");
}

#[test]
fn record_literal_statement_still_parses() {
    let mut env = Environment::new();
    match run(&mut env, "{a: 1, b: 2}") {
        Ok(Value::Record { fields }) => assert_eq!(fields.len(), 2),
        other => panic!("{other:?}"),
    }
}
