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
fn an_early_return_is_rejected() {
    // A `return` that is not the final statement (would need control
    // flow) is rejected by lower_expr's catch-all.
    let e = lower_err("def u:f(x) { return x; x + 1 }\nu:f(1)");
    assert!(
        matches!(e, LowerError::Unsupported(ref m) if m.contains("Return")),
        "expected an Unsupported(Return...) error, got {e:?}"
    );
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
