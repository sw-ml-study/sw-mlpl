//! Regression for issue #6 / C2: newlines inside an array literal `[ ]`
//! are insignificant, so a matrix can be formatted across lines.

use mlpl_parser::{Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    assert_eq!(stmts.len(), 1);
    stmts.into_iter().next().unwrap()
}

fn array_len(e: Expr) -> usize {
    match e {
        Expr::ArrayLit(elems, _) => elems.len(),
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn newline_after_comma_in_array() {
    assert_eq!(array_len(parse_one("[1, 2,\n3, 4]")), 4);
}

#[test]
fn matrix_formatted_across_lines() {
    let src = "[\n1, 0, 0,\n0, 1, 0,\n0, 0, 1\n]";
    assert_eq!(array_len(parse_one(src)), 9);
}

#[test]
fn single_line_array_still_parses() {
    assert_eq!(array_len(parse_one("[1, 2, 3]")), 3);
}
