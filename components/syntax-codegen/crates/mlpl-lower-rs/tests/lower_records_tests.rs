//! Records + Results lowering (compile-to-rust records-results step).
//!
//! Record literals lower to `CVal::record`; a numeric field access
//! extracts the field and unwraps its numeric payload. `ok`/`err`
//! lower to `CVal::result`. (The `?`/`check` propagation operator,
//! which needs function-returns-CVal, is a separate rung.)

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

// -- Record literals --

#[test]
fn record_literal_lowers_to_cval_record() {
    let s = lower_src("{X: 1, Y: 2}").unwrap();
    assert!(s.contains("CVal :: record"), "{s}");
    assert!(s.contains("\"X\""), "{s}");
    assert!(s.contains("\"Y\""), "{s}");
}

#[test]
fn record_numeric_field_wraps_as_arr() {
    let s = lower_src("{X: 1}").unwrap();
    assert!(s.contains("CVal :: Arr"), "{s}");
}

#[test]
fn record_string_field_stays_str() {
    let s = lower_src("{x: \"hi\"}").unwrap();
    assert!(s.contains("CVal :: Str"), "{s}");
}

#[test]
fn nested_record_field_is_a_record() {
    let s = lower_src("{outer: {inner: 7}}").unwrap();
    assert!(s.matches("CVal :: record").count() >= 2, "{s}");
}

// -- Field access --

#[test]
fn field_access_extracts_numeric_field() {
    let s = lower_src("{X: 42, Y: 7}.X").unwrap();
    assert!(s.contains(". field ("), "{s}");
    assert!(s.contains("\"X\""), "{s}");
    assert!(s.contains(". arr ()"), "{s}");
}

#[test]
fn field_access_on_ident_record() {
    let s = lower_src("r = {X: 5}\nr.X").unwrap();
    assert!(s.contains("let mut r ="), "{s}");
    assert!(s.contains(". field ("), "{s}");
}

// -- ok / err Result constructors --

#[test]
fn ok_builtin_lowers_to_result_true() {
    let s = lower_src("ok(5)").unwrap();
    assert!(s.contains("CVal :: result"), "{s}");
    assert!(s.contains("true"), "{s}");
}

#[test]
fn err_builtin_lowers_to_result_false() {
    let s = lower_src("err(5)").unwrap();
    assert!(s.contains("CVal :: result"), "{s}");
    assert!(s.contains("false"), "{s}");
}

// -- The ? / check propagation operator is not lowered yet --

#[test]
fn check_operator_is_unsupported_for_now() {
    // `expr?` desugars to `check(expr)`; propagation needs
    // function-returns-CVal, which this rung does not add.
    let err = lower_src("ok(5)?").unwrap_err();
    assert!(
        matches!(err, LowerError::Unsupported(ref m) if m.contains("check")),
        "got {err:?}"
    );
}
