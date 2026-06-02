//! Regression for issue #6 / C3: `if cond { body }` without an `else`
//! is valid (statement-position if). Previously errored at the trailing
//! newline ("unexpected token 'newline'").

use mlpl_parser::{Expr, lex, parse};

#[test]
fn bare_if_without_else_parses() {
    let stmts = parse(&lex("if gt(1, 0) { 5 }").unwrap()).unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        Expr::If {
            then_body,
            else_body,
            ..
        } => {
            assert_eq!(then_body.len(), 1);
            assert!(else_body.is_empty(), "no else => empty else_body");
        }
        other => panic!("expected If, got {other:?}"),
    }
}

#[test]
fn bare_if_followed_by_more_statements() {
    let stmts = parse(&lex("if gt(1, 0) { 5 }\nx = 7\nx").unwrap()).unwrap();
    assert_eq!(stmts.len(), 3);
}

#[test]
fn if_else_still_parses() {
    let stmts = parse(&lex("if gt(1, 0) { 5 } else { 6 }").unwrap()).unwrap();
    match &stmts[0] {
        Expr::If { else_body, .. } => assert_eq!(else_body.len(), 1),
        other => panic!("expected If, got {other:?}"),
    }
}
