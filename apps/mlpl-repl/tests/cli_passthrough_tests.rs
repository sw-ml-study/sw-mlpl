//! `mlpl-repl -f script.mlpl -- foo bar` CLI passthrough.
//! Saga 31 step 003. Spawns the actual binary because the
//! script -> args() roundtrip is the user-facing contract; the
//! in-process tests in crates/mlpl-eval/tests/args_builtin_tests.rs
//! cover the eval-side side of the contract.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    // Cargo sets `CARGO_BIN_EXE_<bin>` for integration tests; this
    // is the canonical way to find the binary path that matches
    // the current build profile.
    PathBuf::from(env!("CARGO_BIN_EXE_mlpl-repl"))
}

fn write_script(name: &str, body: &str) -> PathBuf {
    let dir = env::temp_dir();
    let path = dir.join(format!("mlpl-cli-passthrough-{name}.mlpl"));
    std::fs::write(&path, body).expect("write tmp script");
    path
}

#[test]
fn trailing_args_after_double_dash_reach_args_builtin() {
    let script = write_script("trailing", "n = list_len(args())\nprint(n)\n");
    let out = Command::new(binary_path())
        .arg("-f")
        .arg(&script)
        .arg("--")
        .arg("alpha")
        .arg("beta")
        .arg("gamma")
        .output()
        .expect("spawn mlpl-repl");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "mlpl-repl failed: stdout={stdout} stderr={stderr}"
    );
    // The script prints n (the list_len), which should be 3.
    // print() emits the scalar's display form on its own line;
    // the eval's final value (also 3) is echoed by the REPL at
    // the end, so we look for "3" anywhere in stdout.
    assert!(
        stdout.contains("3"),
        "expected '3' in stdout, got: {stdout}"
    );
}

#[test]
fn args_can_be_indexed_via_list_get() {
    let script = write_script("indexed", "print(unwrap(list_get(args(), 1)))\n");
    let out = Command::new(binary_path())
        .arg("-f")
        .arg(&script)
        .arg("--")
        .arg("first")
        .arg("second")
        .arg("third")
        .output()
        .expect("spawn mlpl-repl");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "mlpl-repl failed: stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("second"),
        "expected 'second' in stdout, got: {stdout}"
    );
}

#[test]
fn no_trailing_args_yields_empty_strlist() {
    // Without `--` the script's args() returns an empty StrList.
    // list_len of an empty StrList is 0.
    let script = write_script("empty", "print(list_len(args()))\n");
    let out = Command::new(binary_path())
        .arg("-f")
        .arg(&script)
        .output()
        .expect("spawn mlpl-repl");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "mlpl-repl failed: stdout={stdout} stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("0"),
        "expected '0' in stdout, got: {stdout}"
    );
}

#[test]
fn numeric_args_round_trip_through_to_number() {
    // End-to-end: the canonical scripting recipe.
    let script = write_script(
        "numeric",
        "epochs = unwrap(to_number(unwrap(list_get(args(), 0))))\n\
         lr = unwrap(to_number(unwrap(list_get(args(), 1))))\n\
         print(epochs * 1000 + lr)\n",
    );
    let out = Command::new(binary_path())
        .arg("-f")
        .arg(&script)
        .arg("--")
        .arg("100")
        .arg("0.001")
        .output()
        .expect("spawn mlpl-repl");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stdout={stdout}");
    // 100 * 1000 + 0.001 = 100000.001
    assert!(
        stdout.contains("100000.001"),
        "expected '100000.001' in stdout, got: {stdout}"
    );
}
