//! Infix comparison operators (`< > <= >= == !=`): lexing, mapping to
//! `BinOpKind`, and precedence relative to arithmetic (Tier-1 language
//! compactness). Comparisons bind LOOSER than `+ - * /`, so
//! `a + b > c` parses as `(a + b) > c`.

use mlpl_parser::{BinOpKind, Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected one statement from {src:?}");
    stmts.into_iter().next().unwrap()
}

fn top_op(src: &str) -> BinOpKind {
    match parse_one(src) {
        Expr::BinOp { op, .. } => op,
        other => panic!("expected a BinOp from {src:?}, got {other:?}"),
    }
}

#[test]
fn each_operator_maps_to_its_binopkind() {
    assert_eq!(top_op("a < b"), BinOpKind::Lt);
    assert_eq!(top_op("a > b"), BinOpKind::Gt);
    assert_eq!(top_op("a <= b"), BinOpKind::Le);
    assert_eq!(top_op("a >= b"), BinOpKind::Ge);
    assert_eq!(top_op("a == b"), BinOpKind::Eq);
    assert_eq!(top_op("a != b"), BinOpKind::Ne);
}

#[test]
fn eqeq_is_not_two_assignments() {
    // `==` must lex as one comparison token, never two `=`.
    assert_eq!(top_op("x == 5"), BinOpKind::Eq);
}

#[test]
fn comparison_binds_looser_than_arithmetic() {
    // `1 + 2 > 3` -> Gt(Add(1, 2), 3): the `+` groups first.
    match parse_one("1 + 2 > 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Gt);
            assert!(
                matches!(
                    *lhs,
                    Expr::BinOp {
                        op: BinOpKind::Add,
                        ..
                    }
                ),
                "lhs of `>` should be the `+` subexpression"
            );
            assert!(matches!(*rhs, Expr::IntLit(3, _)));
        }
        other => panic!("expected a top-level Gt, got {other:?}"),
    }
}

#[test]
fn comparison_rhs_also_groups_arithmetic() {
    // `a > b + c` -> Gt(a, Add(b, c)).
    match parse_one("a > b + c") {
        Expr::BinOp { op, rhs, .. } => {
            assert_eq!(op, BinOpKind::Gt);
            assert!(matches!(
                *rhs,
                Expr::BinOp {
                    op: BinOpKind::Add,
                    ..
                }
            ));
        }
        other => panic!("expected Gt, got {other:?}"),
    }
}

#[test]
fn multiply_still_binds_tighter_than_compare() {
    // `2 * 3 == 6` -> Eq(Mul(2, 3), 6).
    match parse_one("2 * 3 == 6") {
        Expr::BinOp { op, lhs, .. } => {
            assert_eq!(op, BinOpKind::Eq);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Mul,
                    ..
                }
            ));
        }
        other => panic!("expected Eq, got {other:?}"),
    }
}
