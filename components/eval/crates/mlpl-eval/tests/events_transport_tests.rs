//! The host transport for typed test events: emit_test_event
//! serializes each validated event as ONE JSON line -- to the
//! `test_events_out` file (script mode, synchronous append) or
//! the `test_event_lines` buffer (connect mode drains it per
//! eval). Ordered, exact text, deterministic field order.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

#[test]
fn buffer_mode_collects_ordered_json_lines() {
    let mut env = Environment::new();
    env.test_event_lines = Some(Vec::new());
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"test_start\", suite: \"deques\", name: \"empty deque\"})",
    )
    .unwrap();
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"test_end\", suite: \"deques\", name: \"empty deque\", status: \"passed\", line: 12})",
    )
    .unwrap();
    let lines = env.test_event_lines.as_ref().unwrap();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines[0],
        r#"{"kind":"test_start","name":"empty deque","suite":"deques","version":1}"#
    );
    assert_eq!(
        lines[1],
        r#"{"kind":"test_end","line":12,"name":"empty deque","status":"passed","suite":"deques","version":1}"#
    );
}

#[test]
fn file_mode_appends_one_line_per_event() {
    let dir = std::env::temp_dir().join(format!("mlpl-events-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("events.jsonl");
    let _ = std::fs::remove_file(&path);
    let mut env = Environment::new();
    env.test_events_out = Some(path.clone());
    for kind in ["suite_start", "suite_end"] {
        eval_value(
            &mut env,
            &format!(
                "emit_test_event({{version: 1, kind: \"{kind}\", suite: \"s\", name: \"n\"}})"
            ),
        )
        .unwrap();
    }
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("suite_start"), "{text}");
    assert!(lines[1].contains("suite_end"), "{text}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn strings_escape_exactly_and_kinds_of_values_serialize() {
    let mut env = Environment::new();
    env.test_event_lines = Some(Vec::new());
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"output\", suite: \"s\", name: \"say \\\"hi\\\"\", tags: [\"a\", \"b\"], nested: {note: \"深\"}})",
    )
    .unwrap();
    let line = &env.test_event_lines.as_ref().unwrap()[0];
    assert!(line.contains(r#""name":"say \"hi\"""#), "{line}");
    assert!(line.contains(r#""tags":["a","b"]"#), "{line}");
    assert!(
        line.contains(r#""nested":{"note":"深"}"#),
        "unicode preserved: {line}"
    );
}

#[test]
fn invalid_events_write_nothing() {
    let mut env = Environment::new();
    env.test_event_lines = Some(Vec::new());
    eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"exploded\", suite: \"s\", name: \"n\"})",
    )
    .unwrap_err();
    assert!(env.test_event_lines.as_ref().unwrap().is_empty());
}

#[test]
fn unwritable_path_is_a_runner_error() {
    let mut env = Environment::new();
    env.test_events_out = Some(std::path::PathBuf::from("/nonexistent-dir-xyz/e.jsonl"));
    let e = eval_value(
        &mut env,
        "emit_test_event({version: 1, kind: \"output\", suite: \"s\", name: \"n\"})",
    )
    .unwrap_err();
    assert!(
        e.contains("test-events") || e.contains("nonexistent"),
        "{e}"
    );
}
