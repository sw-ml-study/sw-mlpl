//! Terminal-REPL colon-forms trichotomy: `:disp(x)` is a builtin
//! CALL and must evaluate (it previously fell into the command
//! handler and reported "Unknown command"); `:disp x` is neither a
//! call nor a command and must get the builtin-REFERENCE hint the
//! server gives, keeping all three surfaces in lock-step.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_lines(input: &str) -> (String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mlpl-repl"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn colon_call_evaluates_as_builtin_call() {
    let (stdout, stderr) = run_lines("x = [1, 2, 3]\n:disp(x)\n");
    assert!(
        stdout.contains("rank 1"),
        "disp box expected, got stdout {stdout:?} stderr {stderr:?}"
    );
    assert!(
        !stderr.contains("Unknown command"),
        "colon call must not hit the command handler: {stderr:?}"
    );
}

#[test]
fn colon_builtin_with_space_gets_the_trichotomy_hint() {
    let (_, stderr) = run_lines("x = [1, 2, 3]\n:disp x\n");
    assert!(
        stderr.contains("builtin REFERENCE"),
        "trichotomy hint expected, got {stderr:?}"
    );
}
