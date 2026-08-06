//! expunge(name | [names]) -- APL's quad-EX as a builtin -- and
//! the `:erase` REPL command ()ERASE lineage): 1 means the name
//! is FREE afterwards (idempotent), 0 means malformed; every
//! value table, the u: function table, and the @test registry
//! all let go.

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
fn variables_of_every_kind_are_expunged() {
    let mut env = Environment::new();
    eval_value(&mut env, "x = [1, 2, 3]").unwrap();
    eval_value(&mut env, "s = \"hello\"").unwrap();
    eval_value(&mut env, "r = {a: 1}").unwrap();
    for name in ["x", "s", "r"] {
        assert_eq!(scalar(&mut env, &format!("expunge(\"{name}\")")), 1.0);
        let e = eval_value(&mut env, name).unwrap_err();
        assert!(
            e.contains("undefined") || e.contains("not bound") || e.contains("nknown"),
            "{name} should be gone: {e}"
        );
    }
    // Re-binding after expunge works.
    eval_value(&mut env, "x = 7").unwrap();
    assert_eq!(scalar(&mut env, "x"), 7.0);
}

#[test]
fn user_functions_and_their_test_rows_are_expunged() {
    let mut env = Environment::new();
    eval_value(&mut env, "@test\ndef u:probe() { 1 }").unwrap();
    let v = eval_value(&mut env, "tests()").unwrap();
    assert!(matches!(&v, Value::StrList { items } if items == &["probe".to_string()]));
    assert_eq!(scalar(&mut env, "expunge(\"u:probe\")"), 1.0);
    let e = eval_value(&mut env, "u:probe()").unwrap_err();
    assert!(e.contains("undefined"), "{e}");
    let v = eval_value(&mut env, "tests()").unwrap();
    assert!(
        matches!(&v, Value::StrList { items } if items.is_empty()),
        "the registry must let go too: {v:?}"
    );
}

#[test]
fn expunge_is_idempotent_and_list_form_returns_a_mask() {
    let mut env = Environment::new();
    eval_value(&mut env, "x = 1").unwrap();
    // x bound, y never bound, "9bad" malformed -> [1, 1, 0].
    let v = eval_value(&mut env, "expunge([\"x\", \"y\", \"9bad\"])").unwrap();
    assert!(
        matches!(&v, Value::Array(a) if a.data() == [1.0, 1.0, 0.0]),
        "{v:?}"
    );
    // Expunging again: still free -> 1.
    assert_eq!(scalar(&mut env, "expunge(\"x\")"), 1.0);
}

#[test]
fn malformed_names_return_zero_not_errors() {
    let mut env = Environment::new();
    for bad in ["\"\"", "\"a b\"", "\"1x\"", "\"u:\"", "\"a-b\""] {
        assert_eq!(scalar(&mut env, &format!("expunge({bad})")), 0.0, "{bad}");
    }
    // A non-string argument IS an error (wrong type, not a name).
    let e = eval_value(&mut env, "expunge(42)").unwrap_err();
    assert!(e.contains("expunge"), "{e}");
}

#[test]
fn erase_command_removes_multiple_names() {
    let mut env = Environment::new();
    eval_value(&mut env, "x = 1").unwrap();
    eval_value(&mut env, "def u:f() { 2 }").unwrap();
    let out = mlpl_eval::inspect(&mut env, ":erase x u:f").expect(":erase is a command");
    assert!(out.contains("x") && out.contains("u:f"), "{out}");
    assert!(eval_value(&mut env, "x").is_err());
    assert!(eval_value(&mut env, "u:f()").is_err());
    // Expressions rejected loudly, like the other name commands.
    let out = mlpl_eval::inspect(&mut env, ":erase x + y").unwrap();
    assert!(out.contains("names, not expressions"), "{out}");
}
