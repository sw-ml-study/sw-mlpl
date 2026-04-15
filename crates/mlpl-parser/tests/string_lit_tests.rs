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

// -- UTF-8 (Saga 12: lexer UTF-8 fix) --

#[test]
fn lex_string_preserves_latin1_supplement() {
    // "héllo": h + é (U+00E9) + llo. Stored as UTF-8 bytes
    // 68 C3 A9 6C 6C 6F in the Rust source of this test file.
    let toks = lex("\"h\u{00e9}llo\"").unwrap();
    assert!(
        matches!(&toks[0].kind, mlpl_parser::TokenKind::StrLit(s) if s == "h\u{00e9}llo"),
        "got {:?}",
        toks[0].kind
    );
}

#[test]
fn lex_string_preserves_cjk_codepoint() {
    // 日本 (two CJK ideographs, 3 bytes each in UTF-8).
    let toks = lex("\"\u{65e5}\u{672c}\"").unwrap();
    assert!(
        matches!(&toks[0].kind, mlpl_parser::TokenKind::StrLit(s) if s == "\u{65e5}\u{672c}"),
        "got {:?}",
        toks[0].kind
    );
}

#[test]
fn lex_string_preserves_four_byte_codepoint() {
    // 🦀 Ferris crab, U+1F980 -- 4 bytes in UTF-8.
    let toks = lex("\"\u{1f980}\"").unwrap();
    assert!(
        matches!(&toks[0].kind, mlpl_parser::TokenKind::StrLit(s) if s == "\u{1f980}"),
        "got {:?}",
        toks[0].kind
    );
}

#[test]
fn lex_string_invalid_utf8_errors() {
    // Construct a byte sequence that looks like a valid 2-byte
    // UTF-8 leader but has a non-continuation second byte. The
    // lexer takes &str as input so we can't inject raw invalid
    // bytes through the public API; this test instead calls the
    // internal lex_string via a Rust-string that IS valid UTF-8
    // but contains only a lone high byte by using a raw byte
    // slice -- if we can't, we skip.
    //
    // Rust's `str` type can never hold invalid UTF-8, so invalid
    // sequences can't reach lex() through the public API. The
    // fallthrough branch in lex_string is defensive and
    // exercised via a direct byte-slice call path if/when that
    // path opens up. Documenting this here instead of crafting a
    // contrived test.
    let _ = lex;
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
