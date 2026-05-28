use mlpl_parser::{BinOpKind, Expr, lex, parse};

fn parse_one(src: &str) -> Expr {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1, "expected 1 stmt, got {}", stmts.len());
    stmts.into_iter().next().unwrap()
}

// -- Function calls --

#[test]
fn fn_call_one_arg() {
    match parse_one("shape(x)") {
        Expr::FnCall { name, args, .. } => {
            assert_eq!(name, "shape");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::Ident(_, _)));
        }
        other => panic!("expected FnCall, got {other:?}"),
    }
}

#[test]
fn fn_call_two_args() {
    match parse_one("reshape(x, [2, 2])") {
        Expr::FnCall { name, args, .. } => {
            assert_eq!(name, "reshape");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::Ident(_, _)));
            assert!(matches!(args[1], Expr::ArrayLit(_, _)));
        }
        other => panic!("expected FnCall, got {other:?}"),
    }
}

#[test]
fn fn_call_int_arg() {
    match parse_one("iota(6)") {
        Expr::FnCall { name, args, .. } => {
            assert_eq!(name, "iota");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::IntLit(6, _)));
        }
        other => panic!("expected FnCall, got {other:?}"),
    }
}

#[test]
fn fn_call_nested() {
    match parse_one("shape(reshape(x, [2, 3]))") {
        Expr::FnCall { name, args, .. } => {
            assert_eq!(name, "shape");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::FnCall { .. }));
        }
        other => panic!("expected FnCall, got {other:?}"),
    }
}

#[test]
fn fn_call_no_args() {
    // Not in syntax spec but should parse gracefully
    match parse_one("foo()") {
        Expr::FnCall { name, args, .. } => {
            assert_eq!(name, "foo");
            assert!(args.is_empty());
        }
        other => panic!("expected FnCall, got {other:?}"),
    }
}

// -- Assignment --

#[test]
fn assign_int() {
    match parse_one("x = 42") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(*value, Expr::IntLit(42, _)));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn assign_array() {
    match parse_one("x = [1, 2, 3]") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(*value, Expr::ArrayLit(_, _)));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn assign_binop() {
    match parse_one("x = 1 + 2") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(
                *value,
                Expr::BinOp {
                    op: BinOpKind::Add,
                    ..
                }
            ));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn assign_fn_call() {
    match parse_one("x = reshape(y, [2, 3])") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(*value, Expr::FnCall { .. }));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

// -- Disambiguation --

#[test]
fn bare_ident() {
    match parse_one("x") {
        Expr::Ident(name, _) => assert_eq!(name, "x"),
        other => panic!("expected Ident, got {other:?}"),
    }
}

#[test]
fn ident_in_arithmetic() {
    // x + 1 should parse x as Ident, not trigger assignment
    match parse_one("x + 1") {
        Expr::BinOp { op, lhs, .. } => {
            assert_eq!(op, BinOpKind::Add);
            assert!(matches!(*lhs, Expr::Ident(_, _)));
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

// -- Multi-statement --

#[test]
fn multi_assign() {
    let tokens = lex("x = 1\ny = x + 1").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 2);
    assert!(matches!(stmts[0], Expr::Assign { .. }));
    assert!(matches!(stmts[1], Expr::Assign { .. }));
}

// -- Axis-label annotation on assignment (Saga 11.5 Phase 2) --

#[test]
fn assign_with_axis_annotation_desugars_to_label_call() {
    match parse_one("x : [batch, dim] = reshape(iota(6), [2, 3])") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "x");
            match *value {
                Expr::FnCall {
                    name: fname, args, ..
                } => {
                    assert_eq!(fname, "label");
                    assert_eq!(args.len(), 2);
                    assert!(matches!(args[0], Expr::FnCall { .. }));
                    match &args[1] {
                        Expr::ArrayLit(elems, _) => {
                            assert_eq!(elems.len(), 2);
                            assert!(matches!(&elems[0], Expr::StrLit(s, _) if s == "batch"));
                            assert!(matches!(&elems[1], Expr::StrLit(s, _) if s == "dim"));
                        }
                        other => panic!("expected ArrayLit of StrLit, got {other:?}"),
                    }
                }
                other => panic!("expected FnCall(label, ...), got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn assign_with_single_axis_annotation() {
    match parse_one("v : [seq] = iota(5)") {
        Expr::Assign { name, value, .. } => {
            assert_eq!(name, "v");
            match *value {
                Expr::FnCall {
                    name: fname, args, ..
                } => {
                    assert_eq!(fname, "label");
                    match &args[1] {
                        Expr::ArrayLit(elems, _) => {
                            assert_eq!(elems.len(), 1);
                            assert!(matches!(&elems[0], Expr::StrLit(s, _) if s == "seq"));
                        }
                        other => panic!("expected ArrayLit, got {other:?}"),
                    }
                }
                other => panic!("expected FnCall, got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn assign_with_empty_axis_annotation_scalar() {
    match parse_one("s : [] = 42") {
        Expr::Assign { value, .. } => match *value {
            Expr::FnCall { name, args, .. } => {
                assert_eq!(name, "label");
                match &args[1] {
                    Expr::ArrayLit(elems, _) => assert!(elems.is_empty()),
                    other => panic!("expected empty ArrayLit, got {other:?}"),
                }
            }
            other => panic!("expected FnCall, got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}
