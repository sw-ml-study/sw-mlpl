//! Parser tests for record literals and field access.
//! Saga 29 step 001.

use mlpl_parser::{Expr, ParseError, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 stmt, got {stmts:?}");
    stmts.into_iter().next().unwrap()
}

fn parse_err(src: &str) -> ParseError {
    let tokens = lex(src).unwrap();
    parse(&tokens).unwrap_err()
}

#[test]
fn empty_record() {
    match parse_one("{}") {
        Expr::RecordLit { fields, .. } => assert!(fields.is_empty()),
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn single_field_record() {
    match parse_one("{X: 1}") {
        Expr::RecordLit { fields, .. } => {
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, "X");
            assert!(matches!(fields[0].1, Expr::IntLit(1, _)));
        }
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn three_field_record_keeps_source_order() {
    match parse_one("{X: 1, Y: 2, Z: 3}") {
        Expr::RecordLit { fields, .. } => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].0, "X");
            assert_eq!(fields[1].0, "Y");
            assert_eq!(fields[2].0, "Z");
        }
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn trailing_comma_allowed() {
    match parse_one("{X: 1, Y: 2,}") {
        Expr::RecordLit { fields, .. } => assert_eq!(fields.len(), 2),
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn nested_record_field_value() {
    match parse_one("{outer: {inner: 7}}") {
        Expr::RecordLit { fields, .. } => {
            assert_eq!(fields.len(), 1);
            let (name, inner) = &fields[0];
            assert_eq!(name, "outer");
            assert!(matches!(inner, Expr::RecordLit { .. }));
        }
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn array_value_inside_record() {
    match parse_one("{xs: [1, 2, 3]}") {
        Expr::RecordLit { fields, .. } => {
            assert!(matches!(fields[0].1, Expr::ArrayLit(_, _)));
        }
        other => panic!("expected RecordLit, got {other:?}"),
    }
}

#[test]
fn field_access_on_ident() {
    match parse_one("r.x") {
        Expr::FieldAccess {
            receiver, field, ..
        } => {
            assert_eq!(field, "x");
            assert!(matches!(*receiver, Expr::Ident(_, _)));
        }
        other => panic!("expected FieldAccess, got {other:?}"),
    }
}

#[test]
fn field_access_chained() {
    match parse_one("a.b.c") {
        Expr::FieldAccess {
            receiver, field, ..
        } => {
            assert_eq!(field, "c");
            match *receiver {
                Expr::FieldAccess { field, .. } => assert_eq!(field, "b"),
                other => panic!("expected nested FieldAccess, got {other:?}"),
            }
        }
        other => panic!("expected FieldAccess, got {other:?}"),
    }
}

#[test]
fn field_access_on_fn_call() {
    // f(x).y parses as (f(x)).y
    match parse_one("f(x).y") {
        Expr::FieldAccess {
            receiver, field, ..
        } => {
            assert_eq!(field, "y");
            assert!(matches!(*receiver, Expr::FnCall { .. }));
        }
        other => panic!("expected FieldAccess, got {other:?}"),
    }
}

#[test]
fn field_access_on_record_literal() {
    // {x: 7}.x parses as a FieldAccess on a record literal
    match parse_one("{x: 7}.x") {
        Expr::FieldAccess {
            receiver, field, ..
        } => {
            assert_eq!(field, "x");
            assert!(matches!(*receiver, Expr::RecordLit { .. }));
        }
        other => panic!("expected FieldAccess, got {other:?}"),
    }
}

#[test]
fn field_access_binds_tighter_than_add() {
    // r.x + 1 parses as (r.x) + 1
    match parse_one("r.x + 1") {
        Expr::BinOp { lhs, .. } => assert!(matches!(*lhs, Expr::FieldAccess { .. })),
        other => panic!("expected BinOp at top, got {other:?}"),
    }
}

#[test]
fn duplicate_field_name_errors() {
    let err = parse_err("{X: 1, X: 2}");
    assert!(
        matches!(err, ParseError::DuplicateRecordField { ref name, .. } if name == "X"),
        "expected DuplicateRecordField, got {err:?}"
    );
}

#[test]
fn malformed_field_missing_value_errors() {
    // `{X: }` -- nothing after the colon
    assert!(parse(&lex("{X: }").unwrap()).is_err());
}

#[test]
fn malformed_field_missing_colon_errors() {
    // `{X 1}` -- no colon between name and value
    assert!(parse(&lex("{X 1}").unwrap()).is_err());
}

#[test]
fn record_assignment_round_trip() {
    // r = {X: 1, Y: 2} is a valid assignment statement
    match parse_one("r = {X: 1, Y: 2}") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "r");
            assert!(matches!(*value, Expr::RecordLit { .. }));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn repeat_block_still_parses_as_block_not_record() {
    // Sanity: `repeat 3 { x = 1 }` -- the `{` after `repeat` is a
    // block, not a record. This is the block-vs-record disambiguation
    // case: blocks only appear after the repeat/train/for/experiment/
    // device keywords; record literals appear at expression position.
    match parse_one("repeat 3 { x = 1 }") {
        Expr::Repeat { .. } => {}
        other => panic!("expected Repeat, got {other:?}"),
    }
}

#[test]
fn record_inside_repeat_body_parses() {
    // `repeat 3 { r = {X: 1} }` -- the inner `{X: 1}` is a record.
    let tokens = lex("repeat 3 { r = {X: 1} }").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], Expr::Repeat { .. }));
}
