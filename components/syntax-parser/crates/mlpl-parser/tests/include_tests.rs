//! The `include "path"` top-level declaration: contextual (no
//! reserved word), literal-path only, statement position of a
//! source file only.

use mlpl_parser_ast::Expr;

fn parse(src: &str) -> Result<Vec<Expr>, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    mlpl_parser::parse(&toks).map_err(|e| e.to_string())
}

#[test]
fn top_level_include_parses_to_the_node() {
    let stmts = parse("include \"vector.mlpl\"\nx = 1\n").unwrap();
    assert_eq!(stmts.len(), 2);
    let Expr::Include(path, _) = &stmts[0] else {
        panic!("expected Include, got {:?}", stmts[0]);
    };
    assert_eq!(path, "vector.mlpl");
}

#[test]
fn semicolon_and_multiple_includes_are_fine() {
    let stmts = parse("include \"a.mlpl\";\ninclude \"b.mlpl\"\n").unwrap();
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&stmts[0], Expr::Include(p, _) if p == "a.mlpl"));
    assert!(matches!(&stmts[1], Expr::Include(p, _) if p == "b.mlpl"));
}

#[test]
fn include_stays_a_legal_variable_name() {
    let stmts = parse("include = 5\ninclude + 1\n").unwrap();
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&stmts[0], Expr::Assign { .. }));
}

#[test]
fn nested_include_gets_the_top_level_error() {
    let err = parse("repeat 2 { include \"a.mlpl\" }\n").unwrap_err();
    assert!(err.contains("top level"), "{err}");
    let err = parse("def u:f() { include \"a.mlpl\" }\n").unwrap_err();
    assert!(err.contains("top level"), "{err}");
}

#[test]
fn non_literal_argument_is_not_an_include() {
    // `include foo` is NOT the include pattern: it parses exactly
    // as before this feature -- two identifier statements (which
    // then fail at eval as undefined variables). Only the
    // literal-string form is an include; dynamic includes do not
    // exist.
    let stmts = parse("include foo\n").unwrap();
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&stmts[0], Expr::Ident(n, _) if n == "include"));
}
