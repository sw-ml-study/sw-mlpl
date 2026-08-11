//! Lowering `def u:` user functions (compiler-functions, param-only
//! slice): a nested `fn user_<name>` + `u:name(args)` call routing,
//! `return`, and rejection of global reads.

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lowered(src: &str) -> String {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    lower(&stmts).expect("lower").to_string()
}

fn lower_err(src: &str) -> LowerError {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    lower(&stmts).expect_err("expected a lower error")
}

#[test]
fn user_fn_lowers_to_a_nested_fn_and_a_call() {
    let s = lowered("def u:add3(a, b, c) { a + b + c }\nu:add3(1, 2, 3)");
    assert!(s.contains("fn user_add3"), "no nested fn: {s}");
    assert!(s.contains("user_add3 ("), "no call: {s}");
}

#[test]
fn user_fn_with_a_doc_string_and_trailing_return_lowers() {
    // A leading doc-string statement is discarded; a TRAILING
    // `return x + 1` is the body value (lowered as the block tail).
    let s = lowered("def u:inc(x) { \"add one\"; return x + 1 }\nu:inc(5)");
    assert!(s.contains("fn user_inc"), "{s}");
    assert!(s.contains("user_inc ("), "{s}");
}

#[test]
fn if_else_lowers_to_a_rust_if_expression() {
    // `if` is truthy on a non-zero scalar; branches lower via the
    // shared body lowering. Works at top level and in a function.
    let top = lowered("if 1 { 42 } else { 0 }");
    assert!(top.contains("if (") && top.contains("!= 0"), "{top}");
    let inner = lowered("def u:pick(c) { if c { 10 } else { 20 } }\nu:pick(1)");
    assert!(
        inner.contains("fn user_pick") && inner.contains("if ("),
        "{inner}"
    );
}

#[test]
fn an_early_return_inside_a_branch_lowers_to_a_real_return() {
    // A `return` inside an `if` branch must exit the enclosing fn --
    // it lowers to a real Rust `return`.
    let s = lowered("def u:f(x) { if x { return x } else { 0 } }\nu:f(1)");
    assert!(s.contains("return"), "no real return: {s}");
}

#[test]
fn user_fn_reading_a_global_is_rejected() {
    // `g` is neither a parameter nor a body-local -> Unsupported.
    let e = lower_err("def u:f(x) { x + g }\nu:f(1)");
    match e {
        LowerError::Unsupported(m) => assert!(m.contains("'g'"), "{m}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
