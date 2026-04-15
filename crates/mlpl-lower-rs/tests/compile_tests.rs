//! End-to-end compile test: lower MLPL source, paste the tokens
//! into a tiny Rust program, invoke `rustc`, run the binary, parse
//! stdout, and assert the numeric result matches the interpreter.
//!
//! Slow (one rustc invocation per test) but proves the whole
//! lowering pipeline is real. Gated behind an env var so CI and
//! day-to-day `cargo test` runs skip it unless explicitly asked.
//! Run with: `MLPL_LOWER_RS_COMPILE_TESTS=1 cargo test -p mlpl-lower-rs`.

use std::path::PathBuf;
use std::process::Command;

use mlpl_lower_rs::lower;
use mlpl_parser::{lex, parse};

fn should_run() -> bool {
    std::env::var("MLPL_LOWER_RS_COMPILE_TESTS").is_ok()
}

fn compile_and_run(src: &str) -> f64 {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    let body = lower(&stmts).expect("lower ok").to_string();

    // Resolve the workspace root so the temp program can point its
    // `mlpl-rt` path dependency at the in-tree crate.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf();

    // Build a tiny Rust program in a fresh temp dir. Using a crate
    // layout (Cargo.toml + src/main.rs) rather than raw rustc so
    // `mlpl-rt`'s transitive deps (mlpl-array, mlpl-core) resolve.
    let tmp = tempdir();
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    let cargo_toml = format!(
        r#"[package]
name = "mlpl_lower_rs_e2e"
edition = "2024"
version = "0.0.0"

[dependencies]
mlpl-rt = {{ path = "{}/crates/mlpl-rt" }}
"#,
        workspace.display()
    );
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml).unwrap();
    let main_rs = format!(
        "fn main() {{\n    let result = {};\n    println!(\"{{}}\", result.data()[0]);\n}}\n",
        body
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs).unwrap();

    let build = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(&tmp)
        .output()
        .expect("cargo build");
    assert!(
        build.status.success(),
        "cargo build failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(tmp.join("target/release/mlpl_lower_rs_e2e"))
        .output()
        .expect("run binary");
    assert!(run.status.success(), "binary exited non-zero");
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout.trim().parse::<f64>().expect("parse f64 stdout")
}

fn tempdir() -> PathBuf {
    // Simple monotonic temp dir under the OS temp root -- avoids a
    // dep on `tempfile` just for one test suite.
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let p = base.join(format!("mlpl-lower-rs-e2e-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn scalar_arithmetic_compiles_and_evaluates() {
    if !should_run() {
        eprintln!("skipping end-to-end compile test; set MLPL_LOWER_RS_COMPILE_TESTS=1 to run");
        return;
    }
    let got = compile_and_run("1 + 2 * 3");
    assert!((got - 7.0).abs() < 1e-9, "expected 7.0, got {got}");
}
