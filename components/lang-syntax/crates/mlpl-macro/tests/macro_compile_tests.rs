//! End-to-end tests for the `mlpl!` proc macro.
//!
//! Writes a tiny cargo project that depends on the `mlpl` facade
//! crate via a path dependency, invokes `cargo build --release`,
//! runs the resulting binary, and checks stdout. Gated behind
//! `MLPL_MACRO_COMPILE_TESTS=1` so the default `cargo test` run
//! stays fast. Run with:
//!
//! ```sh
//! MLPL_MACRO_COMPILE_TESTS=1 cargo test -p mlpl-macro
//! ```
//!
//! This is the same pattern the `mlpl-lower-rs` end-to-end tests
//! use -- keeps the slow path explicit and opt-in.

use std::path::PathBuf;
use std::process::Command;

fn should_run() -> bool {
    std::env::var("MLPL_MACRO_COMPILE_TESTS").is_ok()
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
    let p = base.join(format!("mlpl-macro-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Build and run a tiny mlpl!-using program; return stdout.
fn build_and_run_with_macro_body(body: &str) -> String {
    let ws = workspace_root();
    let tmp = tempdir("e2e");
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    let cargo_toml = format!(
        "[package]\n\
         name = \"mlpl_macro_e2e\"\n\
         edition = \"2024\"\n\
         version = \"0.0.0\"\n\
         \n\
         [dependencies]\n\
         mlpl = {{ path = \"{}/crates/mlpl\" }}\n",
        ws.display()
    );
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml).unwrap();
    let main_rs = format!(
        "use mlpl::mlpl;\n\
         fn main() {{\n\
             let result = mlpl! {{ {body} }};\n\
             println!(\"{{}}\", result.data()[0]);\n\
         }}\n"
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
    let run = Command::new(tmp.join("target/release/mlpl_macro_e2e"))
        .output()
        .expect("run binary");
    assert!(run.status.success(), "binary exited non-zero");
    String::from_utf8(run.stdout).expect("utf8")
}

/// Compile a small mlpl!-using program and capture cargo's stderr,
/// expecting failure. Used for error-surfacing tests.
fn build_expecting_failure(body: &str) -> String {
    let ws = workspace_root();
    let tmp = tempdir("err");
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    let cargo_toml = format!(
        "[package]\n\
         name = \"mlpl_macro_err\"\n\
         edition = \"2024\"\n\
         version = \"0.0.0\"\n\
         \n\
         [dependencies]\n\
         mlpl = {{ path = \"{}/crates/mlpl\" }}\n",
        ws.display()
    );
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml).unwrap();
    let main_rs = format!(
        "use mlpl::mlpl;\n\
         fn main() {{\n\
             let _ = mlpl! {{ {body} }};\n\
         }}\n"
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs).unwrap();
    let build = Command::new("cargo")
        .args(["build", "--quiet"])
        .current_dir(&tmp)
        .output()
        .expect("cargo build");
    assert!(
        !build.status.success(),
        "expected build to fail but it succeeded"
    );
    String::from_utf8_lossy(&build.stderr).to_string()
}

#[test]
fn macro_compiles_scalar_program() {
    if !should_run() {
        eprintln!("skipping end-to-end macro test; set MLPL_MACRO_COMPILE_TESTS=1 to run");
        return;
    }
    let out = build_and_run_with_macro_body("1 + 2 * 3");
    assert_eq!(out.trim(), "7");
}

#[test]
fn macro_compiles_multi_statement_program() {
    if !should_run() {
        eprintln!("skipping end-to-end macro test; set MLPL_MACRO_COMPILE_TESTS=1 to run");
        return;
    }
    // Semicolons separate statements inside the macro because
    // proc_macro tokenization collapses newlines.
    let out = build_and_run_with_macro_body("x = iota(10); reduce_add(x)");
    assert_eq!(out.trim(), "45");
}

#[test]
fn macro_compiles_labeled_matmul_program() {
    if !should_run() {
        eprintln!("skipping end-to-end macro test; set MLPL_MACRO_COMPILE_TESTS=1 to run");
        return;
    }
    // Matching contraction axis labels pass; sum of matmul result
    // should match the interpreter's output.
    let body = "a : [seq, d] = reshape(iota(6) + 1, [2, 3]); \
                b : [d, h] = reshape([1, 0, 0, 1, 1, 1], [3, 2]); \
                reduce_add(matmul(a, b))";
    let out = build_and_run_with_macro_body(body);
    assert_eq!(out.trim(), "30");
}

#[test]
fn macro_mismatched_contraction_labels_is_compile_error() {
    if !should_run() {
        eprintln!("skipping end-to-end macro test; set MLPL_MACRO_COMPILE_TESTS=1 to run");
        return;
    }
    let body = "a : [seq, d] = reshape(iota(6), [2, 3]); \
                b : [time, h] = reshape(iota(12), [3, 4]); \
                matmul(a, b)";
    let stderr = build_expecting_failure(body);
    assert!(
        stderr.contains("mlpl:") && stderr.contains("matmul"),
        "expected mlpl/matmul in compile error output, got:\n{stderr}"
    );
}

#[test]
fn macro_parse_error_surfaces_at_compile_time() {
    if !should_run() {
        eprintln!("skipping end-to-end macro test; set MLPL_MACRO_COMPILE_TESTS=1 to run");
        return;
    }
    // `@` is not an MLPL token -- should fail at lex time.
    let stderr = build_expecting_failure("1 @ 2");
    assert!(
        stderr.contains("mlpl:"),
        "expected mlpl error prefix in stderr:\n{stderr}"
    );
}
