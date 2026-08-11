//! `?` / `check` propagation + CVal function returns.
//!
//! A user function whose body produces `ok`/`err` (or uses `?`)
//! returns `CVal` instead of `DenseArray`. `check(expr)` (the desugar
//! of `expr?`) unwraps an `ok` payload or early-returns the whole
//! `err` Result from the enclosing CVal-returning function.

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

#[test]
fn result_returning_fn_has_cval_signature() {
    // A body that produces `ok`/`err` -> the fn returns CVal.
    let s = lower_src("def u:mk(n) { ok(n) }\nu:mk(3)").unwrap();
    assert!(s.contains("fn user_mk"), "{s}");
    assert!(s.contains("-> :: mlpl_rt :: CVal"), "{s}");
}

#[test]
fn record_returning_fn_has_cval_signature() {
    let s = lower_src("def u:pt(n) { {X: n} }\nu:pt(3).X").unwrap();
    assert!(s.contains("fn user_pt"), "{s}");
    assert!(s.contains("-> :: mlpl_rt :: CVal"), "{s}");
}

#[test]
fn plain_fn_keeps_densearray_signature() {
    // No ok/err/check/record -> stays the DenseArray fast path, and
    // no function in the program returns CVal.
    let s = lower_src("def u:add(a, b) { a + b }\nu:add(1, 2)").unwrap();
    assert!(s.contains("fn user_add"), "{s}");
    assert!(s.contains("-> :: mlpl_rt :: DenseArray"), "{s}");
    assert!(!s.contains("-> :: mlpl_rt :: CVal"), "{s}");
}

#[test]
fn check_unwraps_ok_and_early_returns_err() {
    let s = lower_src(
        "def u:fit(n) { ok(n) }\n\
         def u:run(n) { f = u:fit(n)?; f }\n\
         u:run(0)",
    )
    .unwrap();
    // check -> a match: ok payload is the value, err is returned.
    assert!(s.contains("match"), "{s}");
    assert!(s.contains("ok : true"), "{s}");
    assert!(s.contains("return"), "{s}");
    // Both u:fit and u:run return CVal.
    assert!(
        s.matches("-> :: mlpl_rt :: CVal").count() >= 2,
        "expected two CVal-returning fns, got:\n{s}"
    );
}

#[test]
fn top_level_check_is_unsupported() {
    // `?` outside a Result-returning function has nowhere to
    // propagate to.
    let err = lower_src("ok(5)?").unwrap_err();
    assert!(
        matches!(err, LowerError::Unsupported(ref m) if m.contains("check") || m.contains("?")),
        "got {err:?}"
    );
}
