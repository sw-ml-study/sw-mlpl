//! `@word [payload]` annotations (general namespace; @test is the
//! first consumer): stack onto the following def, payloads are
//! record or string literals, orphans error with the rule.

use mlpl_parser_ast::Expr;

fn parse(src: &str) -> Result<Vec<Expr>, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    mlpl_parser::parse(&toks).map_err(|e| e.to_string())
}

fn def_annotations(e: &Expr) -> &Vec<(String, Option<Expr>)> {
    let Expr::FnDef { annotations, .. } = e else {
        panic!("expected FnDef, got {e:?}")
    };
    annotations
}

#[test]
fn bare_and_record_and_string_payloads_stack_in_order() {
    let stmts = parse(
        "@test\n@formula \"H(p) = -sum(p log p)\"\n@doc {latex: \"x\"}\n\
         def u:entropy(p) { 0 - reduce_add(p * log(p)) }\n",
    )
    .unwrap();
    let anns = def_annotations(&stmts[0]);
    assert_eq!(anns.len(), 3);
    assert_eq!(anns[0].0, "test");
    assert!(anns[0].1.is_none());
    assert_eq!(anns[1].0, "formula");
    assert!(matches!(&anns[1].1, Some(Expr::StrLit(s, _)) if s.contains("H(p)")));
    assert_eq!(anns[2].0, "doc");
    assert!(matches!(&anns[2].1, Some(Expr::RecordLit { .. })));
}

#[test]
fn test_annotation_with_record_payload() {
    let stmts = parse("@test {name: \"adds\", tags: [\"fast\"]}\ndef u:t() { ok(1) }\n").unwrap();
    let anns = def_annotations(&stmts[0]);
    assert_eq!(anns.len(), 1);
    assert!(matches!(&anns[0].1, Some(Expr::RecordLit { .. })));
}

#[test]
fn unannotated_defs_have_no_annotations() {
    let stmts = parse("def u:f(x) { x }\n").unwrap();
    assert!(def_annotations(&stmts[0]).is_empty());
}

#[test]
fn orphaned_annotations_error_with_the_rule() {
    let err = parse("@test\nx = 1\n").unwrap_err();
    assert!(err.contains("attach to the NEXT"), "{err}");
    let err = parse("@\ndef u:f() { 1 }\n").unwrap_err();
    assert!(err.contains("needs a word"), "{err}");
}
