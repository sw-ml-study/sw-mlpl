//! Saga 31 step 006: in-process arity + validation tests for
//! `read_stdin()`, `read_stdin_lines()`, and `exit(code)`.
//!
//! The end-to-end behaviour (actually reading piped stdin,
//! actually exiting with the chosen code) lives in
//! `apps/mlpl-repl/tests/stdin_exit_err_tests.rs` because it
//! needs a subprocess. This file covers the eval-side
//! validation that does NOT call `std::process::exit` -- arity
//! errors and out-of-range exit codes -- so they can run in
//! the normal cargo-test process.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn err_msg(src: &str) -> String {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env).unwrap_err();
    format!("{err}")
}

#[test]
fn read_stdin_rejects_args() {
    let msg = err_msg("read_stdin(1)");
    assert!(
        msg.contains("read_stdin") && msg.contains("0 arguments"),
        "expected arity error, got: {msg}"
    );
}

#[test]
fn read_stdin_lines_rejects_args() {
    let msg = err_msg("read_stdin_lines(\"x\")");
    assert!(
        msg.contains("read_stdin_lines") && msg.contains("0 arguments"),
        "expected arity error, got: {msg}"
    );
}

#[test]
fn exit_rejects_missing_arg() {
    let msg = err_msg("exit()");
    assert!(
        msg.contains("exit") && msg.contains("1 arguments"),
        "expected arity error, got: {msg}"
    );
}

#[test]
fn exit_rejects_extra_args() {
    let msg = err_msg("exit(0, 1)");
    assert!(
        msg.contains("exit") && msg.contains("1 arguments"),
        "expected arity error, got: {msg}"
    );
}

#[test]
fn exit_rejects_negative_code() {
    let msg = err_msg("exit(-1)");
    assert!(
        msg.contains("exit") && msg.contains("0"),
        "expected range error, got: {msg}"
    );
}

#[test]
fn exit_rejects_code_over_255() {
    let msg = err_msg("exit(256)");
    assert!(
        msg.contains("exit") && msg.contains("255"),
        "expected range error, got: {msg}"
    );
}

#[test]
fn exit_rejects_non_integer() {
    let msg = err_msg("exit(2.5)");
    assert!(
        msg.contains("exit") && (msg.contains("integer") || msg.contains("0")),
        "expected non-integer error, got: {msg}"
    );
}

#[test]
fn exit_rejects_non_scalar() {
    let msg = err_msg("exit([0, 1])");
    assert!(
        msg.contains("exit") && msg.contains("scalar"),
        "expected non-scalar error, got: {msg}"
    );
}
