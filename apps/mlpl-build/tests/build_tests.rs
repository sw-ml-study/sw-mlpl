//! Integration tests for the `mlpl-build` binary.
//!
//! Each test writes an MLPL source file to a temp dir, invokes the
//! compiled `mlpl-build` binary via `cargo run`, and asserts on
//! either the produced binary's output or on the reported error.
//!
//! Gated by `MLPL_BUILD_TESTS=1` because every test case shells out
//! to `cargo build --release` and takes several seconds. Run with:
//!
//! ```sh
//! MLPL_BUILD_TESTS=1 cargo test -p mlpl-build
//! ```

use std::path::PathBuf;
use std::process::Command;

fn should_run() -> bool {
    std::env::var("MLPL_BUILD_TESTS").is_ok()
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn tempdir(tag: &str) -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let p = base.join(format!("mlpl-build-it-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Run `mlpl-build` with the given args, using `cargo run -p` so we
/// exercise the locally-built binary.
fn run_mlpl_build(args: &[&str]) -> std::process::Output {
    let ws = workspace_root();
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "mlpl-build", "--"])
        .current_dir(&ws);
    for a in args {
        cmd.arg(a);
    }
    cmd.output().expect("run mlpl-build")
}

#[test]
fn builds_native_binary_that_prints_reduce_add() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("reduce");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "reduce_add(iota(10))\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert!(run.status.success(), "produced binary exited non-zero");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "45");
}

#[test]
fn parse_error_reports_source_location_not_rustc_cascade() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("parse-err");
    let src_path = tmp.join("bad.mlpl");
    // `@` is not a valid MLPL character -- the eager lex check
    // should catch this before we ever shell to cargo.
    std::fs::write(&src_path, "1 @ 2\n").unwrap();
    let out_path = tmp.join("bad");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(!result.status.success(), "expected failure");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("mlpl-build:") && stderr.contains("bad.mlpl"),
        "error should mention mlpl-build and source path, got:\n{stderr}"
    );
}

#[test]
fn wasm_target_produces_wasm_output() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    // Skip if the wasm target isn't installed on this machine --
    // we don't want the test suite to fail on a clean dev env.
    let check = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output();
    match check {
        Ok(out) if String::from_utf8_lossy(&out.stdout).contains("wasm32-unknown-unknown") => {}
        _ => {
            eprintln!("skipping wasm test: wasm32-unknown-unknown target not installed");
            return;
        }
    }

    let tmp = tempdir("wasm");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "reduce_add(iota(5))\n").unwrap();
    let out_path = tmp.join("prog.wasm");
    let result = run_mlpl_build(&[
        src_path.to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
        "--target",
        "wasm32-unknown-unknown",
    ]);
    assert!(
        result.status.success(),
        "wasm mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    // The WASM magic number is 0x00 0x61 0x73 0x6d ("\0asm").
    let bytes = std::fs::read(&out_path).expect("read wasm output");
    assert!(
        bytes.starts_with(&[0x00, 0x61, 0x73, 0x6d]),
        "output does not start with WASM magic bytes"
    );
}
