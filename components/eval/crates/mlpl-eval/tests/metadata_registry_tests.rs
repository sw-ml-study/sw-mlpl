//! @test registration at definition time: source order, defaults,
//! field validation, duplicate diagnostics, replace-in-place.

use mlpl_eval::Environment;
use mlpl_eval::env_api::*;

fn eval_in(env: &mut Environment, src: &str) -> Result<(), String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    // Real surfaces (script, serve, wasm) set pending_source; the
    // line stamp comes from it.
    env.set_pending_source(Some(src.to_string()));
    let r = mlpl_eval::eval_program(&stmts, env);
    env.set_pending_source(None);
    r.map(|_| ()).map_err(|e| e.to_string())
}

#[test]
fn registration_is_source_ordered_with_defaults_and_fields() {
    let mut env = Environment::new();
    eval_in(
        &mut env,
        "@test\ndef u:zeta() { ok(1) }\n\
         @test {name: \"alpha case\", tags: [\"fast\", \"math\"], skip: \"wasm\", timeout_ms: 500}\n\
         def u:alpha() { ok(1) }\n\
         def u:not_a_test() { 1 }\n",
    )
    .unwrap();
    assert_eq!(env.tests.len(), 2, "only @test registers");
    assert_eq!(
        env.tests[0].name, "zeta",
        "default name, SOURCE order first"
    );
    assert_eq!(env.tests[0].fn_name, "u:zeta");
    assert_eq!(env.tests[0].source, "repl");
    assert_eq!(env.tests[1].name, "alpha case");
    assert_eq!(env.tests[1].tags, ["fast", "math"]);
    assert_eq!(env.tests[1].skip, "wasm");
    assert_eq!(env.tests[1].timeout_ms, 500.0);
    assert_eq!(env.tests[1].line, 4, "line of the def statement");
}

#[test]
fn duplicate_names_are_loud_and_redefinition_replaces_in_place() {
    let mut env = Environment::new();
    eval_in(&mut env, "@test {name: \"same\"}\ndef u:a() { ok(1) }").unwrap();
    let e = eval_in(&mut env, "@test {name: \"same\"}\ndef u:b() { ok(1) }").unwrap_err();
    assert!(e.contains("duplicate test name"), "{e}");
    assert!(
        e.contains("u:a") && e.contains("u:b"),
        "names both defs: {e}"
    );
    // Re-defining u:a keeps its slot and updates metadata.
    eval_in(
        &mut env,
        "@test {name: \"same\", tags: [\"v2\"]}\ndef u:a() { ok(2) }",
    )
    .unwrap();
    assert_eq!(env.tests.len(), 1);
    assert_eq!(env.tests[0].tags, ["v2"]);
}

#[test]
fn malformed_metadata_is_loud() {
    let mut env = Environment::new();
    let e = eval_in(&mut env, "@test {nope: 1}\ndef u:x() { ok(1) }").unwrap_err();
    assert!(e.contains("unknown or mistyped field `nope`"), "{e}");
    let e = eval_in(&mut env, "@test \"just a string\"\ndef u:y() { ok(1) }").unwrap_err();
    assert!(e.contains("must be a record literal"), "{e}");
}

#[test]
fn non_test_annotations_are_preserved_not_interpreted() {
    let mut env = Environment::new();
    eval_in(
        &mut env,
        "@formula \"H(p) = -sum(p log p)\"\ndef u:entropy(p) { 0 - reduce_add(p * log(p)) }",
    )
    .unwrap();
    assert!(env.tests.is_empty());
    let f = env.get_fn("u:entropy").expect("registered fn");
    assert_eq!(f.annotations.len(), 1);
    assert_eq!(f.annotations[0].0, "formula");
}
