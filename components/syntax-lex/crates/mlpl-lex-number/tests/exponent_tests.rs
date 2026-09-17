//! RS5 (../reasoning-from-scratch): scientific-notation numeric literals.
//! The number lexer accepts an optional `e`/`E` exponent with optional
//! sign; a bare `e` with no following digit is not consumed.

use mlpl_lex_number::lex_number;
use mlpl_lexer_token::TokenKind;

fn float_of(s: &str) -> (f64, usize) {
    match lex_number(s.as_bytes(), 0) {
        Some((TokenKind::FloatLit(v), pos)) => (v, pos),
        other => panic!("expected FloatLit for {s:?}, got {other:?}"),
    }
}

#[test]
fn lexes_scientific_forms() {
    for (src, want) in [
        ("1e-4", 1e-4),
        ("1.5e3", 1.5e3),
        ("2E-10", 2E-10),
        ("6.02e23", 6.02e23),
        ("1e10", 1e10),
        ("-3e2", -3e2),
    ] {
        let (v, pos) = float_of(src);
        assert!(
            (v - want).abs() <= want.abs() * 1e-12 + 1e-18,
            "{src}: got {v}"
        );
        assert_eq!(pos, src.len(), "{src}: consumed the whole literal");
    }
}

#[test]
fn bare_e_is_not_an_exponent() {
    // `1e` -> IntLit(1), leaving `e` for the identifier lexer.
    assert!(matches!(
        lex_number(b"1e", 0),
        Some((TokenKind::IntLit(1), 1))
    ));
    // `1e-` with no digit -> IntLit(1) as well.
    assert!(matches!(
        lex_number(b"1e-", 0),
        Some((TokenKind::IntLit(1), 1))
    ));
}

#[test]
fn plain_int_and_float_unchanged() {
    assert!(matches!(
        lex_number(b"42", 0),
        Some((TokenKind::IntLit(42), 2))
    ));
    assert!(matches!(
        lex_number(b"3.14", 0),
        Some((TokenKind::FloatLit(_), 4))
    ));
}
