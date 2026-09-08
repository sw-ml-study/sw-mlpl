//! Recursion-depth cap: runaway user-function recursion must raise a
//! catchable MLPL error, NOT overflow the stack and abort the process /
//! browser session (upstream-asks #8, mlpl-blockers B6).
//!
//! These run on an explicit large-stack thread. A DEBUG test build has
//! far larger per-frame stack use than the optimized release binary the
//! cap (1000) is calibrated for -- so the harness needs generous room to
//! reach the cap before overflowing. 256 MB is ample; the release binary
//! stays well under 8 MB at the same depth.

use std::thread;

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program_value(&stmts, &mut Environment::new())
}

fn on_big_stack(src: &'static str) -> Result<Value, EvalError> {
    thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || eval(src))
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn runaway_recursion_errors_instead_of_overflowing() {
    let src =
        "def u:c(i, a) { \"count\" if lt(i, 3000) { u:c(i + 1, a + 1) } else { a } }\nu:c(0, 0)";
    let err = on_big_stack(src).expect_err("deep recursion must error, not abort");
    let msg = format!("{err}");
    assert!(msg.contains("recursion too deep"), "{msg}");
    assert!(msg.contains("u:c"), "error should name the function: {msg}");
}

#[test]
fn recursion_within_the_cap_still_succeeds() {
    let src =
        "def u:c(i, a) { \"count\" if lt(i, 500) { u:c(i + 1, a + 1) } else { a } }\nu:c(0, 0)";
    match on_big_stack(src).unwrap() {
        Value::Array(a) => assert_eq!(a.data(), &[500.0]),
        other => panic!("expected scalar 500, got {other:?}"),
    }
}
