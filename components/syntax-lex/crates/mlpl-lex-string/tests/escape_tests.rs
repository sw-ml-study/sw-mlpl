//! RS-escape (../reasoning-from-scratch): unknown backslash escapes in a
//! string literal are lenient -- the backslash is kept literally so LaTeX
//! and math text (\frac, \in, \boxed, \pi) round-trip. Known escapes
//! (\n \t \r \" \\) are unchanged.

use mlpl_lex_string::lex_string;
use mlpl_lexer_token::TokenKind;

fn lex(s: &str) -> String {
    match lex_string(s.as_bytes(), 0) {
        Ok((TokenKind::StrLit(v), pos)) => {
            assert_eq!(pos, s.len(), "consumed whole literal for {s:?}");
            v
        }
        other => panic!("expected StrLit for {s:?}, got {other:?}"),
    }
}

#[test]
fn unknown_escapes_keep_the_backslash() {
    assert_eq!(lex(r#""a\int b""#), r"a\int b");
    assert_eq!(lex(r#""\frac{1}{2}""#), r"\frac{1}{2}");
    assert_eq!(lex(r#""x \in S""#), r"x \in S");
    assert_eq!(lex(r#""\boxed{25}""#), r"\boxed{25}");
    // The downstream's \; normalization case.
    assert_eq!(lex(r#""a\;b""#), r"a\;b");
}

#[test]
fn known_escapes_unchanged() {
    assert_eq!(lex(r#""a\nb""#), "a\nb");
    assert_eq!(lex(r#""a\tb""#), "a\tb");
    assert_eq!(lex(r#""a\rb""#), "a\rb");
    assert_eq!(lex(r#""a\"b""#), "a\"b");
    assert_eq!(lex(r#""a\\b""#), r"a\b");
}

#[test]
fn multibyte_char_after_backslash_roundtrips() {
    // A non-ASCII char after an unknown backslash escape must not be
    // mangled by a byte->char cast.
    assert_eq!(lex("\"\\\u{00e9}\""), "\\\u{00e9}");
}
