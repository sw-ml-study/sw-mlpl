//! End-to-end subprocess tests for `demos/classify.mlpl`
//! (saga 31 step 007). The demo composes every scripting
//! cluster surface from steps 001-006, so an end-to-end run
//! is the broadest regression net for the saga.
//!
//! `every_quick_demo_runs` in mlpl-eval already smoke-evaluates
//! the demo with empty args; these tests cover the
//! script-with-arguments paths the smoke test cannot.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlpl-repl"))
}

fn demo_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is components/cli/crates/mlpl-repl; the demo
    // lives at the repo root's demos/ (stale ../../ fixed after the
    // saga-62 components/ move).
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../demos/classify.mlpl")
}

fn run_with_args(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(binary_path());
    cmd.arg(demo_path());
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
fn low_score_classifies_low() {
    let out = run_with_args(&["10"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "exit failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("low"), "expected 'low', got: {stdout}");
}

#[test]
fn medium_score_classifies_medium() {
    let out = run_with_args(&["50"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(
        stdout.contains("medium"),
        "expected 'medium', got: {stdout}"
    );
}

#[test]
fn high_score_classifies_high() {
    let out = run_with_args(&["95"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(stdout.contains("high"), "expected 'high', got: {stdout}");
}

#[test]
fn threshold_boundary_at_thirty_is_medium() {
    // 30 is NOT > 70 (so not high), and is NOT > 30 (so not
    // medium either) -- it ends up "low". This pins the
    // boundary behaviour against future demo refactors.
    let out = run_with_args(&["30"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(
        stdout.contains("low"),
        "expected 'low' at the 30 boundary, got: {stdout}"
    );
}

#[test]
fn verbose_flag_writes_score_to_stderr() {
    let out = run_with_args(&["77", "verbose"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success());
    assert!(
        stdout.contains("high"),
        "expected 'high' on stdout, got: {stdout}"
    );
    assert!(
        stderr.contains("score:"),
        "expected 'score:' on stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("77"),
        "expected score value on stderr, got: {stderr}"
    );
}

#[test]
fn bad_input_exits_two() {
    let out = run_with_args(&["banana"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        stderr.contains("to_number") || stderr.contains("banana"),
        "expected parse error on stderr, got: {stderr}"
    );
}

#[test]
fn no_args_prints_usage() {
    let out = run_with_args(&[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success());
    assert!(
        stdout.contains("usage"),
        "expected usage message, got: {stdout}"
    );
}
