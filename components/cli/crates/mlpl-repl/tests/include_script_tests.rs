//! Script-mode include integration: sandbox roots, splice
//! semantics, diagnostics naming the right file, and exit-code
//! parity (upstream contract tests 2-8 at the binary level).

use std::path::{Path, PathBuf};
use std::process::Command;

fn setup(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-include-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_mlpl-repl"))
        .args(args)
        .output()
        .expect("run repl");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn include_resolves_against_the_scripts_own_directory_by_default() {
    let dir = setup("default-root");
    write(&dir, "lib.mlpl", "def u:double(x) { x * 2 }\n");
    let root = write(&dir, "root.mlpl", "include \"lib.mlpl\"\nu:double(21)\n");
    let (stdout, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("42"), "{stdout}");
}

#[test]
fn explicit_source_dir_sandboxes_a_root_script_living_elsewhere() {
    // The mlplunit shape: a combined temp script OUTSIDE the
    // source root, includes resolving AGAINST the root.
    let sources = setup("srcroot");
    write(
        &sources,
        "subject.mlpl",
        "def u:native_double(x) { x * 2 }\n",
    );
    let elsewhere = setup("combined");
    let root = write(
        &elsewhere,
        "combined.mlpl",
        "include \"subject.mlpl\"\nu:native_double(21)\n",
    );
    let (stdout, _, code) = run(&[
        "--source-dir",
        sources.to_str().unwrap(),
        "-f",
        root.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("42"), "{stdout}");
}

#[test]
fn nested_includes_resolve_relative_to_the_including_file() {
    let dir = setup("nested");
    write(&dir, "sub/leaf.mlpl", "def u:leaf() { 7 }\n");
    write(&dir, "sub/mid.mlpl", "include \"leaf.mlpl\"\n");
    let root = write(&dir, "root.mlpl", "include \"sub/mid.mlpl\"\nu:leaf()\n");
    let (stdout, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains('7'), "{stdout}");
}

#[test]
fn traversal_escape_is_rejected_with_the_rule() {
    let dir = setup("escape");
    // The escaping target EXISTS -- the sandbox, not a missing
    // file, must be what rejects it. Root = the script's dir.
    write(&dir, "outside.mlpl", "x = 1\n");
    let root = write(&dir, "sub/root.mlpl", "include \"../outside.mlpl\"\n1\n");
    let (_, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("escapes the source root"), "{stderr}");
}

#[test]
fn parse_errors_name_the_included_file_and_line() {
    let dir = setup("diag");
    write(&dir, "bad.mlpl", "ok_line = 1\nx = = 2\n");
    let root = write(&dir, "root.mlpl", "include \"bad.mlpl\"\n1\n");
    let (_, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("bad.mlpl"), "{stderr}");
    assert!(stderr.contains(":2:"), "line number expected: {stderr}");
}

#[test]
fn cycles_print_the_chain() {
    let dir = setup("cycle");
    write(&dir, "a.mlpl", "include \"b.mlpl\"\n");
    write(&dir, "b.mlpl", "include \"a.mlpl\"\n");
    let root = dir.join("a.mlpl");
    let (_, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("cycle"), "{stderr}");
    assert!(stderr.matches("a.mlpl").count() >= 2, "chain: {stderr}");
}

#[test]
fn final_err_semantics_survive_includes() {
    let dir = setup("exitcode");
    write(&dir, "lib.mlpl", "def u:id(x) { x }\n");
    let root = write(&dir, "root.mlpl", "include \"lib.mlpl\"\nerr(\"boom\")\n");
    let (_, stderr, code) = run(&["-f", root.to_str().unwrap()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("boom"), "{stderr}");
}
