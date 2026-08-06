//! run_script: fresh-environment child execution with
//! structured outcomes and captured typed test events.

use mlpl_cli::script_exec::run_script_value;
use mlpl_eval::{Environment, RunScriptOpts, Value};

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-run-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn field<'a>(v: &'a Value, name: &str) -> &'a Value {
    let Value::Result { ok: true, payload } = v else {
        panic!("expected ok(record): {v:?}")
    };
    let Value::Record { fields } = payload.as_ref() else {
        panic!("expected record payload: {v:?}")
    };
    &fields[name]
}

fn opts(capture: bool) -> RunScriptOpts {
    RunScriptOpts {
        source_dir: None,
        data_dir: None,
        capture,
    }
}

#[test]
fn ok_err_error_and_exit_come_back_as_data() {
    let dir = sandbox("status");
    std::fs::write(dir.join("ok.mlpl"), "ok(41 + 1)\n").unwrap();
    std::fs::write(dir.join("bad.mlpl"), "err(\"nope\")\n").unwrap();
    std::fs::write(dir.join("boom.mlpl"), "take([1, 2], 0, 9)\n").unwrap();
    std::fs::write(dir.join("bye.mlpl"), "exit(3)\n").unwrap();
    let cases = [
        ("ok.mlpl", "ok"),
        ("bad.mlpl", "err"),
        ("boom.mlpl", "error"),
        ("bye.mlpl", "exit"),
    ];
    for (file, status) in cases {
        let v = run_script_value(&dir.join(file), &opts(false));
        assert!(
            matches!(field(&v, "status"), Value::Str(s) if s == status),
            "{file}: {v:?}"
        );
    }
    // The exit case reports its code, and THIS process survived.
    let v = run_script_value(&dir.join("bye.mlpl"), &opts(false));
    assert!(matches!(field(&v, "value"), Value::Str(s) if s == "3"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn definitions_do_not_leak_between_runs_and_events_are_captured() {
    let dir = sandbox("fresh");
    std::fs::write(
        dir.join("first.mlpl"),
        "@test\ndef u:probe() { 1 }\n\
         emit_test_event({version: 1, kind: \"test_end\", suite: \"s\", name: \"probe\", status: \"passed\"})\n",
    )
    .unwrap();
    std::fs::write(dir.join("second.mlpl"), "tests()\n").unwrap();
    let v = run_script_value(&dir.join("first.mlpl"), &opts(true));
    assert!(matches!(field(&v, "events_kind"), Value::Str(s) if s == "test_events"));
    let Value::StrList { items } = field(&v, "events") else {
        panic!("events list: {v:?}")
    };
    assert_eq!(items.len(), 1);
    assert!(items[0].contains("\"status\":\"passed\""), "{items:?}");
    // A later run starts FRESH: no registry carry-over.
    let v = run_script_value(&dir.join("second.mlpl"), &opts(false));
    assert!(
        matches!(field(&v, "value"), Value::Str(s) if !s.contains("probe")),
        "fresh env must not see the earlier @test: {v:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn include_works_inside_the_child() {
    let dir = sandbox("inc");
    std::fs::write(dir.join("helper.mlpl"), "def u:two() { 2 }\n").unwrap();
    std::fs::write(
        dir.join("main.mlpl"),
        "include \"helper.mlpl\"\nok(u:two())\n",
    )
    .unwrap();
    let v = run_script_value(&dir.join("main.mlpl"), &opts(false));
    assert!(
        matches!(field(&v, "status"), Value::Str(s) if s == "ok"),
        "{v:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_builtin_end_to_end_matches_their_fixture_shape() {
    let dir = sandbox("e2e");
    std::fs::write(dir.join("test_case.mlpl"), "ok(1)\n").unwrap();
    let mut env = Environment::new();
    env.fs_root = Some(dir.clone());
    env.run_script_hook = Some(run_script_value);
    let src = "execution = run_script(\"test_case.mlpl\", {source_dir: \".\", capture: 1})?\n\
               equal(execution.events_kind, \"test_events\")";
    let tokens = mlpl_parser::lex(src).unwrap();
    let stmts = mlpl_parser::parse(&tokens).unwrap();
    let v = mlpl_eval::eval_program_value(&stmts, &mut env).unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data() == [1.0]), "{v:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_script_is_an_err_value() {
    let dir = sandbox("miss");
    let v = run_script_value(&dir.join("ghost.mlpl"), &opts(false));
    assert!(matches!(&v, Value::Result { ok: false, .. }), "{v:?}");
    std::fs::remove_dir_all(&dir).ok();
}
