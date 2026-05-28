use mlpl_parser::{Expr, lex, parse};

#[test]
fn def_parses_namespaced_function() {
    let toks = lex("def u:area(r) { r * r }").unwrap();
    let stmts = parse(&toks).unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        Expr::FnDef {
            name, params, body, ..
        } => {
            assert_eq!(name, "u:area");
            assert_eq!(params, &["r"]);
            assert_eq!(body.len(), 1);
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn def_rejects_bare_name() {
    let toks = lex("def area(r) { r }").unwrap();
    let err = parse(&toks).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("namespaced") || msg.contains("unexpected"),
        "expected namespace error, got: {msg}"
    );
}

#[test]
fn def_multi_params() {
    let toks = lex("def math:add(a, b) { a + b }").unwrap();
    let stmts = parse(&toks).unwrap();
    match &stmts[0] {
        Expr::FnDef { name, params, .. } => {
            assert_eq!(name, "math:add");
            assert_eq!(params, &["a", "b"]);
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn def_zero_params() {
    let toks = lex("def u:greet() { 42 }").unwrap();
    let stmts = parse(&toks).unwrap();
    match &stmts[0] {
        Expr::FnDef { params, .. } => assert!(params.is_empty()),
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn return_with_value() {
    let toks = lex("def u:f(x) { return x + 1 }").unwrap();
    let stmts = parse(&toks).unwrap();
    match &stmts[0] {
        Expr::FnDef { body, .. } => match &body[0] {
            Expr::Return { value: Some(_), .. } => {}
            other => panic!("expected Return with value, got {other:?}"),
        },
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn return_bare() {
    let toks = lex("def u:f() { return }").unwrap();
    let stmts = parse(&toks).unwrap();
    match &stmts[0] {
        Expr::FnDef { body, .. } => match &body[0] {
            Expr::Return { value: None, .. } => {}
            other => panic!("expected bare Return, got {other:?}"),
        },
        other => panic!("expected FnDef, got {other:?}"),
    }
}
