use mlpl_core::Span;
use mlpl_parser::{ParseError, TokenKind, lex};

fn kinds(src: &str) -> Vec<TokenKind> {
    lex(src).unwrap().into_iter().map(|t| t.kind).collect()
}

// -- Single tokens --

#[test]
fn integer() {
    let tokens = lex("42").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntLit(42));
    assert_eq!(tokens[0].span, Span::new(0, 2));
    assert_eq!(tokens[1].kind, TokenKind::Eof);
}

#[test]
fn float() {
    let tokens = lex("1.5").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::FloatLit(1.5));
    assert_eq!(tokens[0].span, Span::new(0, 3));
}

#[test]
fn identifier() {
    let tokens = lex("my_var").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Ident("my_var".into()));
    assert_eq!(tokens[0].span, Span::new(0, 6));
}

#[test]
fn punctuation() {
    assert_eq!(kinds("("), vec![TokenKind::LParen, TokenKind::Eof]);
    assert_eq!(kinds(")"), vec![TokenKind::RParen, TokenKind::Eof]);
    assert_eq!(kinds("["), vec![TokenKind::LBracket, TokenKind::Eof]);
    assert_eq!(kinds("]"), vec![TokenKind::RBracket, TokenKind::Eof]);
    assert_eq!(kinds(","), vec![TokenKind::Comma, TokenKind::Eof]);
    assert_eq!(kinds("="), vec![TokenKind::Equals, TokenKind::Eof]);
    assert_eq!(kinds(";"), vec![TokenKind::Semicolon, TokenKind::Eof]);
}

#[test]
fn operators() {
    assert_eq!(kinds("+"), vec![TokenKind::Plus, TokenKind::Eof]);
    assert_eq!(kinds("*"), vec![TokenKind::Star, TokenKind::Eof]);
    assert_eq!(kinds("/"), vec![TokenKind::Slash, TokenKind::Eof]);
}

#[test]
fn minus_as_operator() {
    // After a number or ident, minus is an operator
    assert_eq!(
        kinds("1 - 2"),
        vec![
            TokenKind::IntLit(1),
            TokenKind::Minus,
            TokenKind::IntLit(2),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn newline_token() {
    let tokens = lex("1\n2").unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntLit(1));
    assert_eq!(tokens[1].kind, TokenKind::Newline);
    assert_eq!(tokens[2].kind, TokenKind::IntLit(2));
}

// -- Multi-token expressions --

#[test]
fn simple_addition() {
    assert_eq!(
        kinds("1 + 2"),
        vec![
            TokenKind::IntLit(1),
            TokenKind::Plus,
            TokenKind::IntLit(2),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn array_literal() {
    assert_eq!(
        kinds("[1, 2, 3]"),
        vec![
            TokenKind::LBracket,
            TokenKind::IntLit(1),
            TokenKind::Comma,
            TokenKind::IntLit(2),
            TokenKind::Comma,
            TokenKind::IntLit(3),
            TokenKind::RBracket,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn function_call() {
    assert_eq!(
        kinds("reshape(x, [2, 2])"),
        vec![
            TokenKind::Ident("reshape".into()),
            TokenKind::LParen,
            TokenKind::Ident("x".into()),
            TokenKind::Comma,
            TokenKind::LBracket,
            TokenKind::IntLit(2),
            TokenKind::Comma,
            TokenKind::IntLit(2),
            TokenKind::RBracket,
            TokenKind::RParen,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn assignment() {
    assert_eq!(
        kinds("x = 42"),
        vec![
            TokenKind::Ident("x".into()),
            TokenKind::Equals,
            TokenKind::IntLit(42),
            TokenKind::Eof,
        ]
    );
}

// -- Comments --

#[test]
fn comment_skipped() {
    assert_eq!(
        kinds("1 + 2 # add"),
        vec![
            TokenKind::IntLit(1),
            TokenKind::Plus,
            TokenKind::IntLit(2),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn comment_only_line() {
    assert_eq!(kinds("# just a comment"), vec![TokenKind::Eof]);
}

// -- Negative numbers --

#[test]
fn negative_integer_at_start() {
    assert_eq!(kinds("-3"), vec![TokenKind::IntLit(-3), TokenKind::Eof]);
}

#[test]
fn negative_float() {
    assert_eq!(
        kinds("-0.5"),
        vec![TokenKind::FloatLit(-0.5), TokenKind::Eof]
    );
}

#[test]
fn minus_after_number_is_operator() {
    assert_eq!(
        kinds("5-3"),
        vec![
            TokenKind::IntLit(5),
            TokenKind::Minus,
            TokenKind::IntLit(3),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn negative_after_operator() {
    assert_eq!(
        kinds("1 + -2"),
        vec![
            TokenKind::IntLit(1),
            TokenKind::Plus,
            TokenKind::IntLit(-2),
            TokenKind::Eof,
        ]
    );
}

// -- Error cases --

#[test]
fn invalid_character() {
    let result = lex("1 @ 2");
    assert!(result.is_err());
    match result.unwrap_err() {
        ParseError::UnexpectedCharacter { ch, span } => {
            assert_eq!(ch, '@');
            assert_eq!(span, Span::new(2, 3));
        }
        other => panic!("expected UnexpectedCharacter, got {other:?}"),
    }
}

// -- Span correctness --

#[test]
fn spans_correct() {
    let tokens = lex("x + 10").unwrap();
    assert_eq!(tokens[0].span, Span::new(0, 1)); // x
    assert_eq!(tokens[1].span, Span::new(2, 3)); // +
    assert_eq!(tokens[2].span, Span::new(4, 6)); // 10
}

// -- Whitespace --

#[test]
fn multiple_spaces_and_tabs() {
    assert_eq!(
        kinds("1  \t  2"),
        vec![TokenKind::IntLit(1), TokenKind::IntLit(2), TokenKind::Eof]
    );
}

#[test]
fn semicolon_separator() {
    assert_eq!(
        kinds("x = 1; y = 2"),
        vec![
            TokenKind::Ident("x".into()),
            TokenKind::Equals,
            TokenKind::IntLit(1),
            TokenKind::Semicolon,
            TokenKind::Ident("y".into()),
            TokenKind::Equals,
            TokenKind::IntLit(2),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn empty_input() {
    assert_eq!(kinds(""), vec![TokenKind::Eof]);
}

#[test]
fn underscore_identifier() {
    assert_eq!(
        kinds("_temp"),
        vec![TokenKind::Ident("_temp".into()), TokenKind::Eof]
    );
}
