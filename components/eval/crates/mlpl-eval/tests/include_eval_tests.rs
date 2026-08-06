//! Surfaces WITHOUT a source provider reject `include` with the
//! precise script-mode error (design: docs/static-include-design-mlpl.md).

use mlpl_eval::Environment;

#[test]
fn provider_less_evaluation_rejects_include_precisely() {
    let toks = mlpl_parser::lex("include \"vector.mlpl\"\n").unwrap();
    let stmts = mlpl_parser::parse(&toks).unwrap();
    let mut env = Environment::new();
    let err = mlpl_eval::eval_program(&stmts, &mut env)
        .unwrap_err()
        .to_string();
    assert!(err.contains("script-mode"), "{err}");
    assert!(err.contains("vector.mlpl"), "{err}");
    assert!(err.contains("--source-dir"), "{err}");
}
