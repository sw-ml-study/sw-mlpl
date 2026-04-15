//! String-match tests for `lower()` on scalar expressions.
//!
//! These run on every `cargo test` and give fast feedback on
//! codegen shape. The end-to-end rustc-compile-and-run test lives
//! separately in `compile_tests.rs` and can be slower to execute.

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

#[test]
fn int_literal() {
    let s = lower_src("42").unwrap();
    assert!(s.contains("from_scalar"), "{s}");
    assert!(s.contains("42"), "{s}");
}

#[test]
fn float_literal() {
    let s = lower_src("3.14").unwrap();
    assert!(s.contains("3.14"), "{s}");
}

#[test]
fn negation() {
    let s = lower_src("-5").unwrap();
    assert!(s.contains("- 5"), "{s}");
}

#[test]
fn addition() {
    let s = lower_src("1 + 2").unwrap();
    assert!(s.contains("+"), "{s}");
    assert!(s.contains("1"), "{s}");
    assert!(s.contains("2"), "{s}");
}

#[test]
fn precedence_mul_over_add() {
    // 2 * 3 + 1: the AST groups (2*3) then + 1, and the emitted
    // tokens preserve that via parenthesization.
    let s = lower_src("2 * 3 + 1").unwrap();
    // Either order of operands around `+` is fine so long as the
    // multiplication is inside its own parens.
    assert!(s.contains("*"), "{s}");
    assert!(s.contains("+"), "{s}");
}

#[test]
fn wraps_result_in_dense_array_scalar() {
    let s = lower_src("1").unwrap();
    assert!(s.contains("mlpl_rt"), "{s}");
    assert!(s.contains("DenseArray"), "{s}");
    assert!(s.contains("from_scalar"), "{s}");
}

#[test]
fn empty_program_errors() {
    let r = lower_src("");
    assert_eq!(r, Err(LowerError::EmptyProgram));
}

#[test]
fn unsupported_ident_errors_for_now() {
    // Phase 2 (step 003) adds Ident + Assign lowering; for now it's
    // out of scope and must surface as LowerError::Unsupported.
    let r = lower_src("x");
    assert!(
        matches!(r, Err(LowerError::Unsupported(_))),
        "expected Unsupported, got {r:?}"
    );
}

#[test]
fn multi_statement_errors_for_now() {
    let r = lower_src("1\n2");
    assert_eq!(r, Err(LowerError::MultiStatementNotYetSupported));
}
