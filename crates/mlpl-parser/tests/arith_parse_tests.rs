use mlpl_parser::{BinOpKind, Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    stmts.into_iter().next().unwrap()
}

#[test]
fn simple_add() {
    match parse_one("1 + 2") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(*lhs, Expr::IntLit(1, _)));
            assert!(matches!(*rhs, Expr::IntLit(2, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn mul_higher_precedence() {
    // 1 + 2 * 3 -> Add(1, Mul(2, 3))
    match parse_one("1 + 2 * 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(*lhs, Expr::IntLit(1, _)));
            match *rhs {
                Expr::BinOp { op, lhs, rhs, .. } => {
                    assert_eq!(op, BinOpKind::Mul);
                    assert!(matches!(*lhs, Expr::IntLit(2, _)));
                    assert!(matches!(*rhs, Expr::IntLit(3, _)));
                }
                other => panic!("expected inner BinOp, got {other:?}"),
            }
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn mul_then_add() {
    // 1 * 2 + 3 -> Add(Mul(1, 2), 3)
    match parse_one("1 * 2 + 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Mul,
                    ..
                }
            ));
            assert!(matches!(*rhs, Expr::IntLit(3, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn parens_override_precedence() {
    // (1 + 2) * 3 -> Mul(Add(1, 2), 3)
    match parse_one("(1 + 2) * 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Mul);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Add,
                    ..
                }
            ));
            assert!(matches!(*rhs, Expr::IntLit(3, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn left_associative_sub() {
    // 1 - 2 - 3 -> Sub(Sub(1, 2), 3)
    match parse_one("1 - 2 - 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Sub);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Sub,
                    ..
                }
            ));
            assert!(matches!(*rhs, Expr::IntLit(3, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn left_associative_add() {
    // 1 + 2 + 3 -> Add(Add(1, 2), 3)
    match parse_one("1 + 2 + 3") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Add,
                    ..
                }
            ));
            assert!(matches!(*rhs, Expr::IntLit(3, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn array_add() {
    // [1, 2] + [3, 4]
    match parse_one("[1, 2] + [3, 4]") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(*lhs, Expr::ArrayLit(_, _)));
            assert!(matches!(*rhs, Expr::ArrayLit(_, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn all_four_ops() {
    // Just verify they parse without error
    for op_str in ["+", "-", "*", "/"] {
        let src = format!("1 {op_str} 2");
        let _ = parse_one(&src);
    }
}

#[test]
fn division_precedence() {
    // 6 / 2 + 1 -> Add(Div(6, 2), 1)
    match parse_one("6 / 2 + 1") {
        Expr::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(
                *lhs,
                Expr::BinOp {
                    op: BinOpKind::Div,
                    ..
                }
            ));
            assert!(matches!(*rhs, Expr::IntLit(1, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}
