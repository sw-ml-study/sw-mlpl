//! `while cond { body }` + `break [value]` + `continue`. Saga 31 step 005.
//!
//! Truthiness rules match `if`: scalar non-zero or `Ok(_)` is truthy.
//! Loop value defaults to `0`; `break value` overrides.

use mlpl_eval::{Environment, Value, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn val(src: &str, env: &mut Environment) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn scalar(v: Value) -> f64 {
    match v {
        Value::Array(a) => {
            assert_eq!(a.rank(), 0, "expected scalar, got rank {}", a.rank());
            a.data()[0]
        }
        other => panic!("expected scalar Array, got {other}"),
    }
}

#[test]
fn while_loops_until_cond_falsy() {
    // i counts up to 5, then the cond goes false and the loop
    // exits with the default 0 value.
    let mut env = Environment::new();
    let src = "i = 0\nwhile i - 5 { i = i + 1 }\ni";
    let v = val(src, &mut env);
    assert_eq!(scalar(v), 5.0);
}

#[test]
fn while_returns_zero_when_cond_initially_false() {
    let mut env = Environment::new();
    let v = val("while 0 { 1 }", &mut env);
    assert_eq!(scalar(v), 0.0);
}

#[test]
fn while_loop_value_is_zero_on_normal_exit() {
    // The loop expression itself evaluates to 0 if no break-with-value.
    let mut env = Environment::new();
    let v = val("i = 0\nx = while i - 3 { i = i + 1 }\nx", &mut env);
    assert_eq!(scalar(v), 0.0);
}

#[test]
fn break_with_value_yields_that_value() {
    let mut env = Environment::new();
    let v = val(
        "i = 0\nx = while 1 { i = i + 1\nif i - 3 { 0 } else { break 99 } }\nx",
        &mut env,
    );
    assert_eq!(scalar(v), 99.0);
}

#[test]
fn bare_break_yields_zero() {
    let mut env = Environment::new();
    let v = val("x = while 1 { break }\nx", &mut env);
    assert_eq!(scalar(v), 0.0);
}

#[test]
fn continue_skips_remainder_of_body() {
    // Skip i == 3: iterate i from 1..=5 incrementing s only when i != 3.
    // Final s = 4 (incremented at i = 1, 2, 4, 5).
    let mut env = Environment::new();
    let src = "
i = 0
s = 0
while i - 5 {
  i = i + 1
  if i - 3 { 0 } else { continue }
  s = s + 1
}
s
";
    let v = val(src, &mut env);
    assert_eq!(scalar(v), 4.0);
}

#[test]
fn nested_while_break_only_exits_innermost() {
    let mut env = Environment::new();
    let src = "
outer = 0
while outer - 3 {
  outer = outer + 1
  inner = 0
  while 1 {
    inner = inner + 1
    if inner - 2 { 0 } else { break 0 }
  }
}
outer
";
    let v = val(src, &mut env);
    assert_eq!(scalar(v), 3.0);
}

#[test]
fn while_cond_uses_result_truthiness() {
    // to_number("42") is Ok -> truthy once; we break to avoid infinite loop.
    let mut env = Environment::new();
    let v = val("x = while to_number(\"42\") { break 7 }\nx", &mut env);
    assert_eq!(scalar(v), 7.0);
}

#[test]
fn while_cond_non_scalar_array_errors() {
    let mut env = Environment::new();
    let err = eval_program(
        &parse(&lex("while [1, 2, 3] { 0 }").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("while condition"),
        "expected message about non-scalar cond, got: {msg}"
    );
}

#[test]
fn break_outside_loop_errors() {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex("break").unwrap()).unwrap(), &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("break") && msg.contains("outside"),
        "expected break-outside-loop error, got: {msg}"
    );
}

#[test]
fn continue_outside_loop_errors() {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex("continue").unwrap()).unwrap(), &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("continue") && msg.contains("outside"),
        "expected continue-outside-loop error, got: {msg}"
    );
}

#[test]
fn while_format_round_trip() {
    // Display roundtrip; matches `repeat` / `for` formatting.
    let exprs = parse(&lex("while x { x = x - 1 }").unwrap()).unwrap();
    let rendered = format!("{}", exprs[0]);
    assert_eq!(rendered, "while x { x = (x - 1) }");
}

#[test]
fn break_with_value_format() {
    let exprs = parse(&lex("while 1 { break 42 }").unwrap()).unwrap();
    let rendered = format!("{}", exprs[0]);
    assert_eq!(rendered, "while 1 { break 42 }");
}

#[test]
fn bare_break_format() {
    let exprs = parse(&lex("while 1 { break }").unwrap()).unwrap();
    let rendered = format!("{}", exprs[0]);
    assert_eq!(rendered, "while 1 { break }");
}

#[test]
fn continue_format() {
    let exprs = parse(&lex("while 1 { continue }").unwrap()).unwrap();
    let rendered = format!("{}", exprs[0]);
    assert_eq!(rendered, "while 1 { continue }");
}
