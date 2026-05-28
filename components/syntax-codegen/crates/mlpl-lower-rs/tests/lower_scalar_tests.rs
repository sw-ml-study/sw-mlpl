//! String-match tests for `lower()` on scalar expressions.
//!
//! Every MLPL expression lowers to a Rust expression of type
//! `::mlpl_rt::DenseArray`, so these assertions look for presence
//! of the expected function calls and operators in the emitted
//! token string. The end-to-end compile-and-run proof lives in
//! `compile_tests.rs`.

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
fn negation_goes_through_map() {
    // `-5` folds into an IntLit(-5) at lex time; use `-iota(3)` so
    // the unary-neg AST path is actually exercised.
    let s = lower_src("-iota(3)").unwrap();
    assert!(s.contains(". map"), "{s}");
}

#[test]
fn addition_uses_apply_binop() {
    let s = lower_src("1 + 2").unwrap();
    assert!(s.contains("apply_binop"), "{s}");
    // The closure body contains the operator.
    assert!(s.contains("+"), "{s}");
}

#[test]
fn precedence_mul_over_add_threads_both_closures() {
    let s = lower_src("2 * 3 + 1").unwrap();
    assert!(s.contains("*"), "{s}");
    assert!(s.contains("+"), "{s}");
    // Two apply_binop calls since there are two BinOp nodes.
    assert_eq!(s.matches("apply_binop").count(), 2, "{s}");
}

#[test]
fn wraps_scalar_in_dense_array() {
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
