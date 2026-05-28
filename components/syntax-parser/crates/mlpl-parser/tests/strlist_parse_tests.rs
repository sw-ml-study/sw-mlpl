//! Parser regression tests for the `[...]` array literal when
//! it carries string elements. The parser does not distinguish
//! between numeric and string arrays -- both produce
//! `Expr::ArrayLit` -- but we pin the shape here so a future
//! parser rewrite cannot accidentally drop string elements on
//! the floor.
//! Saga 29 step 002.

use mlpl_parser::{Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 stmt, got {stmts:?}");
    stmts.into_iter().next().unwrap()
}

#[test]
fn string_array_lit_parses_to_array_lit_of_strlit() {
    match parse_one("[\"a\", \"b\", \"c\"]") {
        Expr::ArrayLit(elems, _) => {
            assert_eq!(elems.len(), 3);
            assert!(
                elems.iter().all(|e| matches!(e, Expr::StrLit(_, _))),
                "every element should be a StrLit, got {elems:?}"
            );
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn single_string_array_lit_parses() {
    match parse_one("[\"only\"]") {
        Expr::ArrayLit(elems, _) => {
            assert_eq!(elems.len(), 1);
            assert!(matches!(elems[0], Expr::StrLit(_, _)));
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn mixed_kinds_still_parse_into_a_single_array_lit() {
    // Parser accepts heterogeneous elements; the eval layer is
    // the one that flags the mix. This keeps the parser cheap
    // and the error message tutoring (with the actual kinds the
    // user typed) clear.
    match parse_one("[\"a\", 1]") {
        Expr::ArrayLit(elems, _) => {
            assert_eq!(elems.len(), 2);
            assert!(matches!(elems[0], Expr::StrLit(_, _)));
            assert!(matches!(elems[1], Expr::IntLit(1, _)));
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}
