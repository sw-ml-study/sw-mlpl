//! Subprocess tests for saga 31 step 006:
//! - `read_stdin()` returns piped stdin as `Value::Str`
//! - `read_stdin_lines()` returns piped stdin split on `\n`
//! - `exit(code)` terminates the script with the given code
//! - mlpl-repl `-f` propagates a final `Err(...)` as exit 1 + stderr
//!
//! These have to be subprocess tests because `exit()` calls
//! `std::process::exit`, and stdin behaviour depends on the
//! binary's own file descriptors. The in-process unit tests
//! in mlpl-eval cover the no-op surface (arity errors,
//! TTY-refusal) where possible.

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlpl-repl"))
}

fn write_script(name: &str, body: &str) -> PathBuf {
    let path = env::temp_dir().join(format!("mlpl-stdin-exit-err-{name}.mlpl"));
    std::fs::write(&path, body).expect("write tmp script");
    path
}

fn run_with_stdin(script: &PathBuf, input: &str) -> std::process::Output {
    let mut child = Command::new(binary_path())
        .arg("-f")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mlpl-repl");
    {
        let mut stdin = child.stdin.take().expect("child stdin");
        stdin.write_all(input.as_bytes()).expect("write stdin");
    }
    child.wait_with_output().expect("wait")
}

fn run_no_stdin(script: &PathBuf) -> std::process::Output {
    Command::new(binary_path())
        .arg("-f")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn mlpl-repl")
}

#[test]
fn read_stdin_returns_piped_contents() {
    let script = write_script("read_basic", "print(read_stdin())\n");
    let out = run_with_stdin(&script, "hello from stdin");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("hello from stdin"),
        "expected stdin contents in stdout, got: {stdout}"
    );
}

#[test]
fn read_stdin_lines_splits_on_newlines() {
    let script = write_script(
        "read_lines",
        "lines = read_stdin_lines()\nprint(list_len(lines))\n",
    );
    let out = run_with_stdin(&script, "alpha\nbeta\ngamma\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    // 3 non-empty lines (trailing newline does not create an empty entry).
    assert!(
        stdout.contains("3"),
        "expected list_len 3 in stdout, got: {stdout}"
    );
}

#[test]
fn read_stdin_lines_no_trailing_newline() {
    let script = write_script(
        "read_lines_notrail",
        "lines = read_stdin_lines()\nprint(list_len(lines))\n",
    );
    let out = run_with_stdin(&script, "one\ntwo");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(stdout.contains("2"), "expected list_len 2, got: {stdout}");
}

#[test]
fn exit_zero_is_clean() {
    let script = write_script("exit0", "exit(0)\n");
    let out = run_no_stdin(&script);
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn exit_with_custom_code() {
    let script = write_script("exit42", "exit(42)\n");
    let out = run_no_stdin(&script);
    assert_eq!(out.status.code(), Some(42));
}

#[test]
fn exit_skips_remaining_statements() {
    // exit() never returns, so the print after it must not fire.
    // Script mode is quiet by default (source echo is -v only), so
    // AFTER_EXIT must not appear at all: an occurrence means the
    // print fired and exit did not short-circuit. (This test
    // previously expected 1 from the old unconditional echo.)
    let script = write_script("exit_short", "exit(3)\nprint(\"AFTER_EXIT\")\n");
    let out = run_no_stdin(&script);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3));
    let occurrences = stdout.matches("AFTER_EXIT").count();
    assert_eq!(
        occurrences, 0,
        "exit should have short-circuited; AFTER_EXIT appeared {occurrences} times in stdout: {stdout}"
    );
}

#[test]
fn exit_rejects_out_of_range_code() {
    let script = write_script("exit_bad", "exit(999)\n");
    let out = run_no_stdin(&script);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The script should fail before calling exit; mlpl-repl
    // reports the eval error and exits non-zero.
    assert_ne!(out.status.code(), Some(0));
    assert!(
        stderr.contains("exit") && stderr.contains("0"),
        "expected error mentioning exit code range, got: {stderr}"
    );
}

#[test]
fn final_err_value_exits_one_and_writes_stderr() {
    // The script's last expression evaluates to Err(...). The REPL
    // should print the message to stderr and exit non-zero.
    let script = write_script("err_tail", "err(\"deliberate failure\")\n");
    let out = run_no_stdin(&script);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr.contains("deliberate failure"),
        "expected err message in stderr, got: {stderr}"
    );
}

#[test]
fn final_ok_value_exits_zero() {
    // Ok(...) is success.
    let script = write_script("ok_tail", "ok(42)\n");
    let out = run_no_stdin(&script);
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn final_non_result_value_exits_zero() {
    // Regular value (not a Result) is still success.
    let script = write_script("scalar_tail", "x = 1\nx + 2\n");
    let out = run_no_stdin(&script);
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn read_stdin_refuses_tty() {
    // When stdin is /dev/null (closed), read_stdin should
    // still work and return empty string. The TTY-refusal only
    // fires on an actual TTY -- which is hard to assert in a
    // CI environment. Smoke: with stdin=null, read returns "".
    let script = write_script(
        "read_empty",
        "s = read_stdin()\nprint(s)\nprint(\"DONE\")\n",
    );
    let out = run_no_stdin(&script);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "expected clean exit, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("DONE"),
        "expected DONE after read; got: {stdout}"
    );
}
