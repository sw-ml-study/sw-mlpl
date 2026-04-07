use mlpl_parser::{Expr, TokenKind, lex, parse};

#[test]
fn lex_simple_string() {
    let toks = lex(r#""hello""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s == "hello"));
}

#[test]
fn lex_empty_string() {
    let toks = lex(r#""""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s.is_empty()));
}

#[test]
fn lex_string_with_spaces() {
    let toks = lex(r#""hello world""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s == "hello world"));
}

#[test]
fn lex_string_escaped_quote() {
    let toks = lex(r#""a\"b""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s == "a\"b"));
}

#[test]
fn lex_string_escaped_backslash() {
    let toks = lex(r#""a\\b""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s == "a\\b"));
}

#[test]
fn lex_string_escaped_newline() {
    let toks = lex(r#""a\nb""#).unwrap();
    assert!(matches!(&toks[0].kind, TokenKind::StrLit(s) if s == "a\nb"));
}

#[test]
fn lex_unterminated_string() {
    let result = lex(r#""hello"#);
    assert!(result.is_err());
}

#[test]
fn parse_string_literal() {
    let toks = lex(r#""hello""#).unwrap();
    let exprs = parse(&toks).unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], Expr::StrLit(s, _) if s == "hello"));
}

#[test]
fn parse_string_as_function_arg() {
    let toks = lex(r#"foo("bar")"#).unwrap();
    let exprs = parse(&toks).unwrap();
    assert_eq!(exprs.len(), 1);
    if let Expr::FnCall { name, args, .. } = &exprs[0] {
        assert_eq!(name, "foo");
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0], Expr::StrLit(s, _) if s == "bar"));
    } else {
        panic!("expected FnCall");
    }
}

#[test]
fn parse_mixed_args() {
    let toks = lex(r#"plot([1,2,3], "scatter")"#).unwrap();
    let exprs = parse(&toks).unwrap();
    if let Expr::FnCall { args, .. } = &exprs[0] {
        assert_eq!(args.len(), 2);
        assert!(matches!(&args[0], Expr::ArrayLit(_, _)));
        assert!(matches!(&args[1], Expr::StrLit(s, _) if s == "scatter"));
    } else {
        panic!("expected FnCall");
    }
}
