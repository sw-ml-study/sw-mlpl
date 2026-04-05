use mlpl_parser::{Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 statement, got {}", stmts.len());
    stmts.into_iter().next().unwrap()
}

// -- Literals --

#[test]
fn parse_integer() {
    match parse_one("42") {
        Expr::IntLit(n, _) => assert_eq!(n, 42),
        other => panic!("expected IntLit, got {other:?}"),
    }
}

#[test]
fn parse_float() {
    match parse_one("1.5") {
        Expr::FloatLit(f, _) => assert!((f - 1.5).abs() < f64::EPSILON),
        other => panic!("expected FloatLit, got {other:?}"),
    }
}

#[test]
fn parse_negative_int() {
    match parse_one("-3") {
        Expr::IntLit(n, _) => assert_eq!(n, -3),
        other => panic!("expected IntLit(-3), got {other:?}"),
    }
}

#[test]
fn parse_identifier() {
    match parse_one("my_var") {
        Expr::Ident(name, _) => assert_eq!(name, "my_var"),
        other => panic!("expected Ident, got {other:?}"),
    }
}

// -- Array literals --

#[test]
fn parse_array_flat() {
    match parse_one("[1, 2, 3]") {
        Expr::ArrayLit(elems, _) => {
            assert_eq!(elems.len(), 3);
            assert!(matches!(elems[0], Expr::IntLit(1, _)));
            assert!(matches!(elems[1], Expr::IntLit(2, _)));
            assert!(matches!(elems[2], Expr::IntLit(3, _)));
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn parse_array_nested() {
    match parse_one("[[1, 2], [3, 4]]") {
        Expr::ArrayLit(rows, _) => {
            assert_eq!(rows.len(), 2);
            match &rows[0] {
                Expr::ArrayLit(elems, _) => assert_eq!(elems.len(), 2),
                other => panic!("expected inner ArrayLit, got {other:?}"),
            }
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn parse_empty_array() {
    match parse_one("[]") {
        Expr::ArrayLit(elems, _) => assert!(elems.is_empty()),
        other => panic!("expected empty ArrayLit, got {other:?}"),
    }
}

// -- Parens --

#[test]
fn parse_parens() {
    // (42) should unwrap to just 42
    match parse_one("(42)") {
        Expr::IntLit(n, _) => assert_eq!(n, 42),
        other => panic!("expected IntLit through parens, got {other:?}"),
    }
}

// -- Multi-statement --

#[test]
fn parse_multi_newline() {
    let tokens = lex("1\n2\n3").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 3);
}

#[test]
fn parse_multi_semicolon() {
    let tokens = lex("1; 2; 3").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 3);
}

#[test]
fn parse_empty_input() {
    let tokens = lex("").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert!(stmts.is_empty());
}

#[test]
fn parse_comment_only() {
    let tokens = lex("# nothing").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert!(stmts.is_empty());
}

// -- Error --

#[test]
fn parse_unexpected_rparen() {
    let tokens = lex(")").unwrap();
    assert!(parse(&tokens).is_err());
}

#[test]
fn parse_unclosed_bracket() {
    let tokens = lex("[1, 2").unwrap();
    assert!(parse(&tokens).is_err());
}
