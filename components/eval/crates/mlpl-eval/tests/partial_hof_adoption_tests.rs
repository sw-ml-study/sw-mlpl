//! Partials must be accepted everywhere a function reference is
//! (the shared apply_callable path): the Result combinators and
//! bracket's use/teardown hooks, not just each/table/atop/over.

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

const PRELUDE: &str = "
def u:add(a, b) { a + b }
def u:checked(a, x) { if gt(x, 0) { ok(a + x) } else { err(\"nonpositive\") } }
";

#[test]
fn result_combinators_accept_partials() {
    let mut env = Environment::new();
    eval_value(&mut env, PRELUDE).unwrap();
    eval_value(&mut env, "add5 = call(:u:add, 5)").unwrap();
    // map_ok: apply the partial inside ok.
    assert_eq!(scalar(&mut env, "unwrap(map_ok(add5, ok(3)))"), 8.0);
    // map_ok bypasses err.
    let v = eval_value(&mut env, "err_message(map_ok(add5, err(\"boom\")))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "boom"), "{v:?}");
    // and_then: partial that returns a Result.
    eval_value(&mut env, "chk1 = call(:u:checked, 1)").unwrap();
    assert_eq!(scalar(&mut env, "unwrap(and_then(chk1, ok(4)))"), 5.0);
    let v = eval_value(&mut env, "err_message(and_then(chk1, ok(0)))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "nonpositive"), "{v:?}");
    // or_else: partial recovers from the error PAYLOAD (add5
    // returns a raw number, which is what or_else yields).
    assert_eq!(scalar(&mut env, "or_else(add5, err(10))"), 15.0);
    // ok bypasses or_else entirely (the ok flows through).
    assert_eq!(scalar(&mut env, "unwrap(or_else(add5, ok(2)))"), 2.0);
}

const HOOKS: &str = "
def u:setup_five() { 5 }
def u:use_with_offset(off, fixture) { ok(fixture + off) }
def u:teardown_with_marker(mark, fixture) { ok(fixture + mark) }
";

#[test]
fn bracket_use_and_teardown_accept_partials() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    eval_value(&mut env, "use10 = call(:u:use_with_offset, 10)").unwrap();
    eval_value(&mut env, "cleanup100 = call(:u:teardown_with_marker, 100)").unwrap();
    // setup 5 -> use adds 10 -> ok(15); teardown ran (its ok
    // does not override a successful use).
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(bracket(:u:setup_five, use10, cleanup100))"
        ),
        15.0
    );
}

#[test]
fn bracket_teardown_partial_failure_still_surfaces() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    eval_value(
        &mut env,
        "def u:teardown_fail(mark, fixture) { err(\"leaked\") }\n\
         use10 = call(:u:use_with_offset, 10)\n\
         td = call(:u:teardown_fail, 1)",
    )
    .unwrap();
    let v = eval_value(&mut env, "err_message(bracket(:u:setup_five, use10, td))").unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "leaked"), "{v:?}");
}

#[test]
fn bracket_still_rejects_builtin_hooks() {
    let mut env = Environment::new();
    eval_value(&mut env, HOOKS).unwrap();
    let e = eval_value(
        &mut env,
        "bracket(:u:setup_five, :mean, :u:teardown_with_marker)",
    )
    .unwrap_err();
    assert!(e.contains("bracket") && e.contains("u:"), "{e}");
}
