#[test]
fn builtin_ref_call_parses_as_fncall() {
    // `:disp(g)` is the quoted builtin APPLIED -- identical to
    // disp(g). (It used to parse as two statements: the ref, then
    // the parenthesized arg -- the ":disp(g) says g undefined" bug.)
    let toks = mlpl_parser::lex(":disp(g)").expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    assert_eq!(stmts.len(), 1, "one CALL statement, not ref + paren-expr");
    let mlpl_parser::Expr::FnCall { name, args, .. } = &stmts[0] else {
        panic!("expected FnCall, got {:?}", stmts[0]);
    };
    assert_eq!(name, "disp");
    assert_eq!(args.len(), 1);
}

#[test]
fn bare_builtin_ref_stays_a_reference() {
    let toks = mlpl_parser::lex(":disp").expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    assert!(
        matches!(&stmts[0], mlpl_parser::Expr::BuiltinRef(n, _) if n == "disp"),
        "bare :disp must remain the first-class reference"
    );
}
