//! Parser tests for record destructuring `{a, b: x} = value`.

use mlpl_parser::{Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 stmt, got {stmts:?}");
    stmts.into_iter().next().unwrap()
}

fn bindings(src: &str) -> Vec<(String, String)> {
    match parse_one(src) {
        Expr::Destructure { bindings, .. } => bindings,
        other => panic!("{src}: expected Destructure, got {other:?}"),
    }
}

fn pair(f: &str, v: &str) -> (String, String) {
    (f.to_string(), v.to_string())
}

#[test]
fn same_name_rename_and_trailing_comma() {
    assert_eq!(bindings("{a, b} = r"), vec![pair("a", "a"), pair("b", "b")]);
    assert_eq!(
        bindings("{a, b: x,} = f(1)?"),
        vec![pair("a", "a"), pair("b", "x")]
    );
    assert_eq!(
        bindings("{\n  train,\n  eval: e\n} = r"),
        vec![pair("train", "train"), pair("eval", "e")]
    );
}

#[test]
fn record_literals_are_untouched() {
    for src in ["{a: 1}", "{a: b}", "{a: b} == c", "{}"] {
        assert!(
            !matches!(parse_one(src), Expr::Destructure { .. }),
            "{src} must stay an expression"
        );
    }
}

#[test]
fn display_round_trips() {
    let e = parse_one("{a, b: x} = f(r)");
    assert_eq!(e.to_string(), "{a, b: x} = f(r)");
    assert_eq!(parse_one(&e.to_string()).to_string(), e.to_string());
}
