//! `try { } catch e { }` + `?` propagation (spike step 011;
//! design: docs/option-result-design.md "Control-flow adapters").

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<mlpl_eval::Value, String> {
    let toks = lex(src).map_err(|e| format!("lex {e:?}"))?;
    let prog = parse(&toks).map_err(|e| format!("parse {e:?}"))?;
    eval_program_value(&prog, env).map_err(|e| format!("eval {e:?}"))
}

fn data(env: &mut Environment, src: &str) -> Vec<f64> {
    match run(env, src).expect("eval ok") {
        mlpl_eval::Value::Array(a) => a.data().to_vec(),
        other => panic!("expected array, got {other:?}"),
    }
}

// ---- try/catch ----

#[test]
fn catch_demotes_a_hard_error_to_the_handler_value() {
    let mut env = Environment::new();
    assert_eq!(
        data(&mut env, "try { take([1, 2, 3], 0, 9) } catch e { 42 }"),
        vec![42.0]
    );
}

#[test]
fn ok_body_skips_the_handler() {
    let mut env = Environment::new();
    assert_eq!(data(&mut env, "try { 5 } catch e { 9 }"), vec![5.0]);
}

#[test]
fn handler_binds_the_error_record() {
    let mut env = Environment::new();
    let v = run(&mut env, "try { take([1, 2], 0, 7) } catch e { e.message }").expect("eval");
    match v {
        mlpl_eval::Value::Str(s) => assert!(!s.is_empty(), "message populated: {s:?}"),
        other => panic!("expected string message, got {other:?}"),
    }
    let v = run(&mut env, "try { take([1, 2], 0, 7) } catch e { e.kind }").expect("eval");
    match v {
        mlpl_eval::Value::Str(s) => assert!(!s.is_empty(), "kind populated: {s:?}"),
        other => panic!("expected string kind, got {other:?}"),
    }
}

#[test]
fn err_values_flow_through_uncaught() {
    // err(...) is DATA, not a hard error: the handler must not run.
    let mut env = Environment::new();
    let v = run(&mut env, "try { err(\"boom\") } catch e { 1 }").expect("eval");
    assert!(
        matches!(v, mlpl_eval::Value::Result { ok: false, .. }),
        "{v:?}"
    );
}

#[test]
fn try_is_an_expression() {
    let mut env = Environment::new();
    assert_eq!(
        data(
            &mut env,
            "x = try { take([1, 2], 0, 9) } catch e { fill([2], 0) }; tally(x)"
        ),
        vec![2.0]
    );
}

#[test]
fn multi_statement_bodies_yield_the_last_value() {
    let mut env = Environment::new();
    assert_eq!(
        data(&mut env, "try { a = 1; a + 1 } catch e { 0 }"),
        vec![2.0]
    );
}

// ---- ? / check ----

#[test]
fn question_unwraps_ok_inside_a_fn() {
    let mut env = Environment::new();
    run(&mut env, "def u:inc(r) { v = r?; ok(v + 1) }").expect("def");
    let v = run(&mut env, "u:inc(ok(2))").expect("call");
    match v {
        mlpl_eval::Value::Result { ok: true, payload } => {
            assert_eq!(format!("{payload}"), "3");
        }
        other => panic!("expected Ok(3), got {other:?}"),
    }
}

#[test]
fn question_early_returns_the_err_from_a_fn() {
    let mut env = Environment::new();
    run(&mut env, "def u:inc(r) { v = r?; ok(v + 1) }").expect("def");
    let v = run(&mut env, "u:inc(err(\"no\"))").expect("call");
    assert!(
        matches!(v, mlpl_eval::Value::Result { ok: false, .. }),
        "{v:?}"
    );
}

#[test]
fn pipeline_propagates_the_first_err() {
    let mut env = Environment::new();
    // `/` is float division, so the guard is a range check: halving
    // below 4 "fails" -- enough to exercise Err propagation.
    run(
        &mut env,
        "def u:half(x) { if lt(x, 4) { err(\"too small\") } else { ok(x / 2) } }",
    )
    .expect("def half");
    run(&mut env, "def u:quarter(x) { h = u:half(x)?; u:half(h) }").expect("def quarter");
    let v = run(&mut env, "u:quarter(8)").expect("call");
    assert!(
        matches!(v, mlpl_eval::Value::Result { ok: true, .. }),
        "{v:?}"
    );
    let v = run(&mut env, "u:quarter(6)").expect("call");
    assert!(
        matches!(v, mlpl_eval::Value::Result { ok: false, .. }),
        "{v:?}"
    );
}

#[test]
fn top_level_question_on_err_is_loud() {
    let mut env = Environment::new();
    let msg = run(&mut env, "err(\"boom\")?").expect_err("must be loud");
    assert!(msg.contains("UnwrapOnErr") || msg.contains("boom"), "{msg}");
}

#[test]
fn top_level_question_on_ok_unwraps() {
    let mut env = Environment::new();
    assert_eq!(data(&mut env, "ok(7)?"), vec![7.0]);
}

#[test]
fn check_spelled_form_matches_question() {
    let mut env = Environment::new();
    assert_eq!(data(&mut env, "check(ok(7))"), vec![7.0]);
}

#[test]
fn question_on_non_result_errors() {
    let mut env = Environment::new();
    let msg = run(&mut env, "5?").expect_err("non-Result");
    assert!(!msg.is_empty());
}
