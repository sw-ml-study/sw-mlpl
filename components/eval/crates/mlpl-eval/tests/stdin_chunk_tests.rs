//! `read_stdin_chunk(max_bytes)` interpreter validation (../demo-file-
//! processing bounded-stdin pipeline). These cover the budget-
//! validation path, which returns a clean `err` BEFORE any stdin byte
//! is read -- so they are deterministic without a piped fixture. The
//! actual byte / EOF reading is exercised end-to-end on the
//! compile-to-Rust surface (`mlpl-build` `read_stdin_chunk_*` e2e),
//! which shares the same validation, record, and EOF semantics.

use mlpl_eval::{Environment, Value};

fn eval(src: &str) -> Value {
    let toks = mlpl_parser::lex(src).expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    let mut env = Environment::new();
    mlpl_eval::eval_program_value(&stmts, &mut env).expect("eval")
}

fn assert_err_mentions(src: &str, needle: &str) {
    match eval(src) {
        Value::Result { ok: false, payload } => {
            let Value::Str(msg) = *payload else {
                panic!("err payload must be a string, got {payload:?}");
            };
            assert!(
                msg.contains("read_stdin_chunk") && msg.contains(needle),
                "expected an err naming `{needle}`, got: {msg}"
            );
        }
        other => panic!("expected an err Result, got {other:?}"),
    }
}

#[test]
fn zero_budget_is_err_and_reads_no_stdin() {
    assert_err_mentions("read_stdin_chunk(0)", "positive integer");
}

#[test]
fn negative_budget_is_err() {
    assert_err_mentions("read_stdin_chunk(-4)", "positive integer");
}

#[test]
fn fractional_budget_is_err() {
    assert_err_mentions("read_stdin_chunk(1.5)", "positive integer");
}

#[test]
fn non_scalar_budget_is_err() {
    assert_err_mentions("read_stdin_chunk([4, 8])", "scalar positive integer");
}

#[test]
fn wrong_arity_is_a_hard_error() {
    let toks = mlpl_parser::lex("read_stdin_chunk(1, 2)").expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    let mut env = Environment::new();
    // Arity is a hard EvalError (not a Result), like other builtins.
    assert!(mlpl_eval::eval_program_value(&stmts, &mut env).is_err());
}
