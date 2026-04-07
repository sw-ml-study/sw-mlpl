use mlpl_parser::{Expr, TokenKind, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 statement, got {}", stmts.len());
    stmts.into_iter().next().unwrap()
}

#[test]
fn lex_repeat_keyword() {
    let tokens = lex("repeat").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Repeat);
}

#[test]
fn lex_braces() {
    let tokens = lex("{ }").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LBrace);
    assert_eq!(tokens[1].kind, TokenKind::RBrace);
}

#[test]
fn parse_repeat_simple() {
    let expr = parse_one("repeat 3 { x = x + 1 }");
    match expr {
        Expr::Repeat { count, body, .. } => {
            assert!(matches!(*count, Expr::IntLit(3, _)));
            assert_eq!(body.len(), 1);
            assert!(matches!(&body[0], Expr::Assign { name, .. } if name == "x"));
        }
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn parse_repeat_multi_statement() {
    let expr = parse_one("repeat 10 { x = x + 1; y = y * 2 }");
    match expr {
        Expr::Repeat { body, .. } => {
            assert_eq!(body.len(), 2);
        }
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn parse_repeat_newline_separated() {
    let expr = parse_one("repeat 5 {\nx = x + 1\ny = y - 1\n}");
    match expr {
        Expr::Repeat { body, .. } => {
            assert_eq!(body.len(), 2);
        }
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn parse_repeat_with_expr_count() {
    let expr = parse_one("repeat 2 + 3 { x = 1 }");
    match expr {
        Expr::Repeat { count, .. } => {
            assert!(matches!(*count, Expr::BinOp { .. }));
        }
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn parse_repeat_zero() {
    let expr = parse_one("repeat 0 { x = 1 }");
    match expr {
        Expr::Repeat { count, .. } => {
            assert!(matches!(*count, Expr::IntLit(0, _)));
        }
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn parse_repeat_after_other_stmts() {
    let tokens = lex("x = 0\nrepeat 3 { x = x + 1 }").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&stmts[0], Expr::Assign { .. }));
    assert!(matches!(&stmts[1], Expr::Repeat { .. }));
}
