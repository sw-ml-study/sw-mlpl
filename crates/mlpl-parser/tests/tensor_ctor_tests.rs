use mlpl_parser::{Expr, TensorCtorKind, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    stmts.into_iter().next().unwrap()
}

#[test]
fn param_scalar_shape() {
    match parse_one("param[3]") {
        Expr::TensorCtor { kind, shape, .. } => {
            assert_eq!(kind, TensorCtorKind::Param);
            assert_eq!(shape.len(), 1);
            assert!(matches!(shape[0], Expr::IntLit(3, _)));
        }
        other => panic!("expected TensorCtor, got {other:?}"),
    }
}

#[test]
fn param_matrix_shape() {
    match parse_one("param[2, 3]") {
        Expr::TensorCtor { kind, shape, .. } => {
            assert_eq!(kind, TensorCtorKind::Param);
            assert_eq!(shape.len(), 2);
            assert!(matches!(shape[0], Expr::IntLit(2, _)));
            assert!(matches!(shape[1], Expr::IntLit(3, _)));
        }
        other => panic!("expected TensorCtor, got {other:?}"),
    }
}

#[test]
fn tensor_matrix_shape() {
    match parse_one("tensor[4, 5]") {
        Expr::TensorCtor { kind, shape, .. } => {
            assert_eq!(kind, TensorCtorKind::Tensor);
            assert_eq!(shape.len(), 2);
        }
        other => panic!("expected TensorCtor, got {other:?}"),
    }
}

#[test]
fn tensor_rank3_shape() {
    match parse_one("tensor[2, 3, 4]") {
        Expr::TensorCtor { kind, shape, .. } => {
            assert_eq!(kind, TensorCtorKind::Tensor);
            assert_eq!(shape.len(), 3);
        }
        other => panic!("expected TensorCtor, got {other:?}"),
    }
}

#[test]
fn param_assigned() {
    let tokens = lex("w = param[3, 2]").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "w");
            assert!(matches!(
                **value,
                Expr::TensorCtor {
                    kind: TensorCtorKind::Param,
                    ..
                }
            ));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}
