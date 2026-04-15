//! Tests for :experiments and compare(a, b) (Saga 12 step 008).

use mlpl_eval::{Environment, Value, eval_program, eval_program_value, inspect};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

fn eval_val(src: &str, env: &mut Environment) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = base.join(format!("mlpl-reg-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

// -- :experiments (REPL command) --

#[test]
fn experiments_with_no_runs_reports_empty() {
    let env = Environment::new();
    let out = inspect(&env, ":experiments").unwrap();
    assert!(out.contains("no experiments recorded"), "out: {out}");
}

#[test]
fn experiments_lists_memory_log_entries_in_order() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.5 }", &mut env);
    run("experiment \"b\" { loss_metric = 0.3 }", &mut env);
    let out = inspect(&env, ":experiments").unwrap();
    assert!(out.contains("a"), "missing 'a':\n{out}");
    assert!(out.contains("b"), "missing 'b':\n{out}");
    let pos_a = out.find("a").unwrap();
    let pos_b = out.find("b").unwrap();
    assert!(pos_a < pos_b, "order wrong:\n{out}");
}

#[test]
fn experiments_shows_top_line_metric() {
    let mut env = Environment::new();
    run(
        "experiment \"run1\" { loss_metric = 0.25; accuracy_metric = 0.9 }",
        &mut env,
    );
    let out = inspect(&env, ":experiments").unwrap();
    // Top-line metric = first alphabetically among _metric names.
    // "accuracy_metric" < "loss_metric" lex -> shown first.
    assert!(
        out.contains("accuracy_metric") && out.contains("0.9"),
        "out: {out}"
    );
}

#[test]
fn experiments_no_metrics_falls_back_to_no_metrics_string() {
    let mut env = Environment::new();
    run("experiment \"bare\" { x = 1 }", &mut env);
    let out = inspect(&env, ":experiments").unwrap();
    assert!(out.contains("bare"), "out: {out}");
    assert!(out.contains("(no metrics)"), "out: {out}");
}

#[test]
fn experiments_merges_on_disk_records_in_terminal() {
    let dir = tempdir("merge");
    // Hand-craft a run.json on disk; :experiments should pick it up
    // even though env.experiment_log is empty.
    let run_dir = dir.join("fromdisk").join("123456");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(
        run_dir.join("run.json"),
        r#"{"name":"fromdisk","timestamp_ns":123456,"metrics":{"loss_metric":0.1},"params_snapshot":{}}"#,
    )
    .unwrap();

    let mut env = Environment::new();
    env.set_exp_dir(dir);
    let out = inspect(&env, ":experiments").unwrap();
    assert!(out.contains("fromdisk"), "out: {out}");
    assert!(out.contains("loss_metric"), "out: {out}");
    assert!(out.contains("0.1"), "out: {out}");
}

// -- compare(a, b) builtin --

#[test]
fn compare_returns_string_value() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.5 }", &mut env);
    run("experiment \"b\" { loss_metric = 0.3 }", &mut env);
    let v = eval_val("compare(\"a\", \"b\")", &mut env);
    assert!(matches!(v, Value::Str(_)), "expected Value::Str, got {v:?}");
}

#[test]
fn compare_shows_both_names_and_values() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.5 }", &mut env);
    run("experiment \"b\" { loss_metric = 0.3 }", &mut env);
    let Value::Str(s) = eval_val("compare(\"a\", \"b\")", &mut env) else {
        panic!();
    };
    assert!(s.contains("a"), "{s}");
    assert!(s.contains("b"), "{s}");
    assert!(s.contains("loss_metric"), "{s}");
    assert!(s.contains("0.5"), "{s}");
    assert!(s.contains("0.3"), "{s}");
}

#[test]
fn compare_shows_delta() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.5 }", &mut env);
    run("experiment \"b\" { loss_metric = 0.3 }", &mut env);
    let Value::Str(s) = eval_val("compare(\"a\", \"b\")", &mut env) else {
        panic!();
    };
    // delta = 0.3 - 0.5 = -0.2. The text should mention a signed
    // delta near -0.2.
    assert!(
        s.contains("-0.2") || s.contains("0.2") && s.contains('-'),
        "delta missing:\n{s}"
    );
}

#[test]
fn compare_uses_most_recent_run_per_name() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.9 }", &mut env);
    // Second "a" run with a different metric value -- compare
    // should use the most recent.
    // Small sleep to ensure distinct timestamps even on fast
    // machines.
    std::thread::sleep(std::time::Duration::from_millis(1));
    run("experiment \"a\" { loss_metric = 0.1 }", &mut env);
    run("experiment \"b\" { loss_metric = 0.5 }", &mut env);
    let Value::Str(s) = eval_val("compare(\"a\", \"b\")", &mut env) else {
        panic!();
    };
    assert!(s.contains("0.1"), "should use latest a's loss:\n{s}");
    assert!(
        !s.contains("0.9"),
        "should not mention older a's loss:\n{s}"
    );
}

#[test]
fn compare_missing_name_errors() {
    let mut env = Environment::new();
    run("experiment \"exists\" { loss_metric = 0.5 }", &mut env);
    let r = eval_program(
        &parse(&lex("compare(\"exists\", \"missing\")").unwrap()).unwrap(),
        &mut env,
    );
    assert!(r.is_err(), "expected error for missing name");
}

#[test]
fn compare_bad_arity_errors() {
    let mut env = Environment::new();
    let r = eval_program(&parse(&lex("compare(\"a\")").unwrap()).unwrap(), &mut env);
    assert!(r.is_err());
}

#[test]
fn compare_non_string_arg_errors() {
    let mut env = Environment::new();
    run("experiment \"a\" { loss_metric = 0.5 }", &mut env);
    let r = eval_program(
        &parse(&lex("compare(\"a\", 42)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(r.is_err());
}
