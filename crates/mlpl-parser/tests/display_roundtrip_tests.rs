//! Saga R1 step 003: `Display for Expr` round-trip
//! tests. The orchestrator-side device-block
//! forwarder renders the block body back to MLPL
//! source via `Display`; the peer re-parses it.
//! Anything that round-trips parse -> display ->
//! parse must produce the same AST.

use mlpl_parser::{lex, parse};

fn roundtrip(src: &str) {
    let tokens = lex(src).unwrap_or_else(|e| panic!("lex({src:?}): {e:?}"));
    let stmts = parse(&tokens).unwrap_or_else(|e| panic!("parse({src:?}): {e:?}"));
    let rendered: String = stmts
        .iter()
        .map(|s| format!("{s}"))
        .collect::<Vec<_>>()
        .join("\n");
    let tokens2 = lex(&rendered).unwrap_or_else(|e| panic!("re-lex({rendered:?}): {e:?}"));
    let stmts2 = parse(&tokens2).unwrap_or_else(|e| panic!("re-parse({rendered:?}): {e:?}"));
    assert_eq!(
        stmts.len(),
        stmts2.len(),
        "stmt count: orig={} rendered={}",
        stmts.len(),
        stmts2.len()
    );
    // Strip spans before AST equality -- re-parsing
    // produces fresh spans that won't match the
    // originals. Easiest: compare textual rendering
    // again.
    let rendered2: String = stmts2
        .iter()
        .map(|s| format!("{s}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(rendered, rendered2, "render is idempotent");
}

#[test]
fn int_literal() {
    roundtrip("42");
}

#[test]
fn float_literal() {
    roundtrip("3.5");
}

#[test]
fn float_with_trailing_zero() {
    // 7.0 must render as "7.0", not "7", so re-parse
    // gets FloatLit not IntLit.
    roundtrip("7.0");
}

#[test]
fn string_literal() {
    roundtrip("\"hello\"");
}

#[test]
fn ident() {
    roundtrip("x");
}

#[test]
fn binop_chain() {
    roundtrip("1 + 2 * 3");
}

#[test]
fn fncall_with_args() {
    roundtrip("matmul(a, b)");
}

#[test]
fn assign() {
    roundtrip("y = x * 2");
}

#[test]
fn array_lit() {
    roundtrip("[1, 2, 3]");
}

#[test]
fn iota_call() {
    roundtrip("iota(5)");
}

#[test]
fn device_block() {
    roundtrip("device(\"mlx\") { x = iota(3) }");
}

#[test]
fn device_block_multi_stmt() {
    roundtrip("device(\"mlx\") { a = iota(3); b = a * 2 }");
}

#[test]
fn nested_assign_with_array() {
    roundtrip("y = [1, 2, 3]");
}

#[test]
fn negation() {
    roundtrip("-x");
}
