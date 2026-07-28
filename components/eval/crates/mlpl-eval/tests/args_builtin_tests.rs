//! `args()` builtin. Saga 31 step 003.
//!
//! Returns the trailing CLI args (after the `--` separator) as a
//! `Value::StrList`. Empty list when no args were passed. The
//! Environment carries the args; the REPL CLI parser populates it
//! before evaluating the script. Tests here exercise the
//! in-process path -- the CLI passthrough (`-- foo bar` reaching
//! the script) is exercised by the integration test in
//! apps/mlpl-repl/tests/.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn val(src: &str, env: &mut Environment) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn strlist(v: Value) -> Vec<String> {
    match v {
        Value::StrList { items } => items,
        other => panic!("expected StrList, got {other}"),
    }
}

#[test]
fn args_returns_empty_strlist_by_default() {
    let mut env = Environment::new();
    let v = val("args()", &mut env);
    assert_eq!(strlist(v), Vec::<String>::new());
}

#[test]
fn args_returns_preset_args() {
    let mut env = Environment::new();
    env.set_cli_args(vec!["foo".into(), "bar".into(), "baz".into()]);
    let v = val("args()", &mut env);
    assert_eq!(strlist(v), vec!["foo", "bar", "baz"]);
}

#[test]
fn args_preserves_arg_order() {
    let mut env = Environment::new();
    env.set_cli_args(vec!["z".into(), "a".into(), "m".into()]);
    let v = val("args()", &mut env);
    assert_eq!(strlist(v), vec!["z", "a", "m"]);
}

#[test]
fn args_returns_single_arg() {
    let mut env = Environment::new();
    env.set_cli_args(vec!["only-one".into()]);
    let v = val("args()", &mut env);
    assert_eq!(strlist(v), vec!["only-one"]);
}

#[test]
fn args_composes_with_list_len() {
    // Script pattern: argc-style count via list_len(args()).
    let mut env = Environment::new();
    env.set_cli_args(vec!["a".into(), "b".into(), "c".into()]);
    let v = val("list_len(args())", &mut env);
    let n = match v {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other}"),
    };
    assert_eq!(n, 3.0);
}

#[test]
fn args_composes_with_to_number_via_list_get() {
    // Canonical scripting pattern: parse a numeric CLI arg.
    // `list_get(args(), 0)` returns Ok("100"); unwrap pulls the
    // string out; to_number parses it; unwrap pulls the f64 out.
    // With cli_args = ["100", "0.001"], the chain reads as
    // "epoch count" and "learning rate" respectively.
    let mut env = Environment::new();
    env.set_cli_args(vec!["100".into(), "0.001".into()]);
    let epochs = val("unwrap(to_number(unwrap(list_get(args(), 0))))", &mut env);
    let lr = val("unwrap(to_number(unwrap(list_get(args(), 1))))", &mut env);
    match (epochs, lr) {
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.data()[0], 100.0);
            assert!((b.data()[0] - 0.001).abs() < 1e-9);
        }
        (a, b) => panic!("expected two scalars, got {a} / {b}"),
    }
}

#[test]
fn list_get_ok_in_range() {
    let mut env = Environment::new();
    env.set_cli_args(vec!["alpha".into(), "beta".into()]);
    let v = val("unwrap(list_get(args(), 1))", &mut env);
    match v {
        Value::Str(s) => assert_eq!(s, "beta"),
        other => panic!("expected Str, got {other}"),
    }
}

#[test]
fn list_get_err_out_of_bounds() {
    let mut env = Environment::new();
    env.set_cli_args(vec!["only".into()]);
    let is_ok = val("is_ok(list_get(args(), 5))", &mut env);
    match is_ok {
        Value::Array(a) => assert_eq!(a.data()[0], 0.0),
        other => panic!("expected scalar Array, got {other}"),
    }
    // unwrap_or fallback path -- the canonical pattern for
    // missing args.
    let with_default = val("unwrap_or(list_get(args(), 5), \"default\")", &mut env);
    match with_default {
        Value::Str(s) => assert_eq!(s, "default"),
        other => panic!("expected Str, got {other}"),
    }
}

#[test]
fn list_get_rejects_non_strlist_receiver() {
    let mut env = Environment::new();
    let err = mlpl_eval::eval_program(&parse(&lex("list_get(42, 0)").unwrap()).unwrap(), &mut env)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("list_get") && msg.contains("string-list"),
        "expected non-strlist error, got: {msg}"
    );
}

#[test]
fn args_rejects_arguments() {
    let mut env = Environment::new();
    let err =
        mlpl_eval::eval_program(&parse(&lex("args(0)").unwrap()).unwrap(), &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("args") && msg.contains("expects"),
        "expected arity error for args(0), got: {msg}"
    );
}
