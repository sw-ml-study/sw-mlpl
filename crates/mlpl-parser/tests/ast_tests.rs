use mlpl_core::Span;
use mlpl_parser::{BinOpKind, Expr};

#[test]
fn int_lit_span() {
    let e = Expr::IntLit(42, Span::new(0, 2));
    assert_eq!(e.span(), Span::new(0, 2));
}

#[test]
fn float_lit_span() {
    let e = Expr::FloatLit(1.5, Span::new(0, 3));
    assert_eq!(e.span(), Span::new(0, 3));
}

#[test]
fn ident_span() {
    let e = Expr::Ident("x".into(), Span::new(0, 1));
    assert_eq!(e.span(), Span::new(0, 1));
}

#[test]
fn array_lit_span() {
    let e = Expr::ArrayLit(
        vec![
            Expr::IntLit(1, Span::new(1, 2)),
            Expr::IntLit(2, Span::new(4, 5)),
        ],
        Span::new(0, 6),
    );
    assert_eq!(e.span(), Span::new(0, 6));
}

#[test]
fn binop_span() {
    let e = Expr::BinOp {
        op: BinOpKind::Add,
        lhs: Box::new(Expr::IntLit(1, Span::new(0, 1))),
        rhs: Box::new(Expr::IntLit(2, Span::new(4, 5))),
        span: Span::new(0, 5),
    };
    assert_eq!(e.span(), Span::new(0, 5));
}

#[test]
fn fn_call_span() {
    let e = Expr::FnCall {
        name: "iota".into(),
        args: vec![Expr::IntLit(5, Span::new(5, 6))],
        span: Span::new(0, 7),
    };
    assert_eq!(e.span(), Span::new(0, 7));
}

#[test]
fn assign_span() {
    let e = Expr::Assign {
        name: "x".into(),
        value: Box::new(Expr::IntLit(42, Span::new(4, 6))),
        span: Span::new(0, 6),
    };
    assert_eq!(e.span(), Span::new(0, 6));
}

#[test]
fn binop_kind_debug() {
    // Ensure all variants exist and are debuggable
    let kinds = [
        BinOpKind::Add,
        BinOpKind::Sub,
        BinOpKind::Mul,
        BinOpKind::Div,
    ];
    for k in &kinds {
        let _ = format!("{k:?}");
    }
}

#[test]
fn expr_clone() {
    let e = Expr::IntLit(1, Span::new(0, 1));
    let e2 = e.clone();
    assert_eq!(e.span(), e2.span());
}
