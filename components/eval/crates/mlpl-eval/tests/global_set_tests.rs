//! global_set(name, value) -- the EXPLICIT global-state escape
//! hatch (mlplunit's in-language-event-reporting gate): binding
//! hygiene stays the default, but a spelled-out global write
//! survives the frame restore and propagates to the top level,
//! so a stateful reporter sink can count events.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

#[test]
fn reporter_sink_state_persists_across_emits() {
    // Their in_language_reporter_case pattern, in the accepted
    // public spelling: the sink counts test_end events.
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "count = 0\npassed = 0\n\
         def u:report(event) {\n\
           if equal(event.kind, \"test_end\") {\n\
             global_set(\"count\", count + 1);\n\
             if equal(event.status, \"passed\") { global_set(\"passed\", passed + 1) } else { 0 }\n\
           } else { 0 };\n\
           ok(1)\n\
         }",
    )
    .unwrap();
    eval_value(&mut env, "test_event_sink(:u:report)").unwrap();
    for status in ["passed", "failed", "passed"] {
        eval_value(
            &mut env,
            &format!(
                "emit_test_event({{version: 1, kind: \"test_end\", suite: \"s\", name: \"n\", status: \"{status}\"}})"
            ),
        )
        .unwrap();
    }
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"test_start\", suite: \"s\", name: \"n\"})",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "count"), 3.0, "test_end events counted");
    assert_eq!(scalar(&mut env, "passed"), 2.0, "passes counted");
}

#[test]
fn writes_propagate_through_nested_frames() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "total = 0\n\
         def u:bump() { global_set(\"total\", total + 1) }\n\
         def u:outer() { u:bump(); u:bump(); 0 }",
    )
    .unwrap();
    eval_value(&mut env, "u:outer()").unwrap();
    assert_eq!(scalar(&mut env, "total"), 2.0);
}

#[test]
fn later_rebinds_are_not_clobbered_by_stale_replays() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "x = 0\ndef u:setx() { global_set(\"x\", 1) }\ndef u:noop(v) { v }",
    )
    .unwrap();
    eval_value(&mut env, "u:setx()").unwrap();
    eval_value(&mut env, "x = 5").unwrap();
    // An unrelated later call must not replay the old write.
    eval_value(&mut env, "u:noop(9)").unwrap();
    assert_eq!(scalar(&mut env, "x"), 5.0);
}

#[test]
fn locals_still_do_not_leak_without_global_set() {
    let mut env = Environment::new();
    eval_value(&mut env, "y = 1\ndef u:shadow() { y = 99; y }").unwrap();
    assert_eq!(scalar(&mut env, "u:shadow()"), 99.0);
    assert_eq!(scalar(&mut env, "y"), 1.0, "hygiene unchanged by default");
}

#[test]
fn every_value_kind_and_cross_kind_rebinding_works() {
    let mut env = Environment::new();
    eval_value(
        &mut env,
        "def u:set_all() {\n\
           global_set(\"gs\", \"hello\");\n\
           global_set(\"gr\", {a: 1});\n\
           global_set(\"gl\", [\"x\", \"y\"]);\n\
           global_set(\"gres\", ok(7));\n\
           0\n\
         }",
    )
    .unwrap();
    eval_value(&mut env, "u:set_all()").unwrap();
    assert!(matches!(
        eval_value(&mut env, "gs").unwrap(),
        Value::Str(s) if s == "hello"
    ));
    assert_eq!(scalar(&mut env, "gr.a"), 1.0);
    assert_eq!(scalar(&mut env, "list_len(gl)"), 2.0);
    assert_eq!(scalar(&mut env, "unwrap(gres)"), 7.0);
    // Cross-kind global rebind: string -> array.
    eval_value(&mut env, "def u:flip() { global_set(\"gs\", [1, 2, 3]) }").unwrap();
    eval_value(&mut env, "u:flip()").unwrap();
    assert_eq!(scalar(&mut env, "tally(gs)"), 3.0);
}

#[test]
fn misuse_is_loud() {
    let mut env = Environment::new();
    let e = eval_value(&mut env, "global_set(\"9bad\", 1)").unwrap_err();
    assert!(e.contains("global_set"), "{e}");
    let e = eval_value(&mut env, "global_set(1, 2)").unwrap_err();
    assert!(e.contains("name"), "{e}");
    let e = eval_value(&mut env, "global_set(\"x\")").unwrap_err();
    assert!(e.contains("global_set"), "{e}");
}
