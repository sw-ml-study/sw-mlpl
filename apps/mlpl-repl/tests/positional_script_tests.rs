//! Saga 31 step 007: positional script-path support for
//! `#!/usr/bin/env mlpl-repl` shebang.
//!
//! Today's CLI requires `-f script.mlpl`. The kernel's
//! shebang-line invocation hands `mlpl-repl` the script path
//! as the FIRST positional argument with no preceding `-f`
//! flag. These tests pin the positional convention so
//! shebang scripts work end-to-end.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlpl-repl"))
}

fn write_script(name: &str, body: &str) -> PathBuf {
    let path = env::temp_dir().join(format!("mlpl-positional-{name}.mlpl"));
    std::fs::write(&path, body).expect("write tmp script");
    path
}

fn run_positional(script: &PathBuf, args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(binary_path());
    cmd.arg(script);
    if !args.is_empty() {
        cmd.arg("--");
        for a in args {
            cmd.arg(a);
        }
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn mlpl-repl")
}

#[test]
fn positional_script_path_runs() {
    let script = write_script("simple", "print(\"hello world\")\n");
    let out = run_positional(&script, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("hello world"),
        "expected output, got: {stdout}"
    );
}

#[test]
fn positional_script_with_args_after_separator() {
    let script = write_script("args", "print(list_len(args()))\n");
    let out = run_positional(&script, &["alpha", "beta"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit failed");
    assert!(stdout.contains("2"), "expected list_len 2, got: {stdout}");
}

#[test]
fn positional_script_propagates_err_exit_code() {
    let script = write_script("err", "err(\"bad input\")\n");
    let out = run_positional(&script, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr.contains("bad input"));
}

#[test]
fn positional_script_propagates_exit_code() {
    let script = write_script("exit5", "exit(5)\n");
    let out = run_positional(&script, &[]);
    assert_eq!(out.status.code(), Some(5));
}

#[test]
fn shebang_line_is_ignored() {
    // `#` is a line comment in MLPL, so a leading `#!` line is
    // skipped by the lexer. This pins the contract: scripts
    // with shebang lines run unchanged.
    let script = write_script("shebang", "#!/usr/bin/env mlpl-repl\nprint(\"ran\")\n");
    let out = run_positional(&script, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("ran"),
        "expected 'ran' in output, got: {stdout}"
    );
}

#[test]
fn dash_f_form_still_works() {
    // The `-f` form is the explicit / back-compat path.
    let script = write_script("dashf", "print(\"old\")\n");
    let out = Command::new(binary_path())
        .arg("-f")
        .arg(&script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("old"));
}

#[test]
fn flag_arg_does_not_trigger_positional_path() {
    // Bare `mlpl-repl --version` should NOT try to interpret
    // `--version` as a script path.
    let out = Command::new(binary_path())
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // version banner mentions the crate name
    assert!(stdout.contains("mlpl"));
}

#[test]
fn nonexistent_positional_path_errors() {
    // A positional arg that looks like a path but doesn't
    // exist should produce a clear error. `mlpl-repl
    // /nonexistent.mlpl` -> exit 1, error on stderr.
    let out = Command::new(binary_path())
        .arg("/nonexistent-mlpl-script-9f3.mlpl")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    assert_ne!(out.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("nonexistent-mlpl-script") || stderr.contains("error reading"),
        "expected file error, got: {stderr}"
    );
}
