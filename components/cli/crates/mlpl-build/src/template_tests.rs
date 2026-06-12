//! Saga 75: unit tests for `template::{render_cargo_toml, candidate_names}`.
//!
//! These were originally inline in main.rs, then extracted to keep main.rs
//! under the module-fn-count limit -- but that extraction landed a
//! malformed, unparseable file (a bare `let` at module scope) that dropped
//! five of the six tests; this file restores them. Included via `#[path]`
//! from template.rs, so `super` resolves against the `template` module.

use super::{candidate_names, render_cargo_toml};
use std::path::PathBuf;

/// The `\Users` in a Windows path is the sequence that produced "invalid
/// unicode 8-digit hex code" when emitted as a TOML basic string; assert
/// the line carrying it uses single-quote (literal-string) form.
fn assert_path_line_is_literal(toml: &str) {
    let line = toml
        .lines()
        .find(|l| l.contains(r"\Users"))
        .expect("path line");
    assert!(
        line.contains("'") && !line.contains('"'),
        "the path line still uses double quotes: {line:?}"
    );
}

/// Issue #3 regression: a Windows-style workspace path containing
/// backslashes must be emitted inside a TOML *literal* string so the
/// backslashes survive verbatim and never reach the basic-string escape
/// parser. (Cargo accepts either separator on Windows -- what matters is
/// single quotes, not slashes.)
#[test]
fn render_cargo_toml_handles_windows_backslash_path() {
    let win_path = PathBuf::from(r"C:\Users\bill\Documents\Projects\sw-mlpl");
    let toml = render_cargo_toml(&win_path).expect("render");
    assert!(
        toml.contains(r"path = 'C:\Users\bill\Documents\Projects\sw-mlpl"),
        "Windows backslashes were not preserved verbatim, got:\n{toml}"
    );
    // Must NOT be a double-quoted basic string -- that triggered issue #3.
    assert!(
        !toml.contains(r#"path = "C:\"#),
        "path was emitted as a basic string; TOML would interpret \
         backslashes as escapes. Full output:\n{toml}"
    );
    assert_path_line_is_literal(&toml);
}

#[test]
fn render_cargo_toml_handles_unix_path() {
    let unix_path = PathBuf::from("/Users/mike/github/sw-ml-study/sw-mlpl");
    let toml = render_cargo_toml(&unix_path).expect("render");
    assert!(
        toml.contains("mlpl = { path = '/Users/mike/github/sw-ml-study/sw-mlpl/crates/mlpl' }"),
        "expected unix-path literal, got:\n{toml}"
    );
}

#[test]
fn render_cargo_toml_rejects_path_with_single_quote() {
    let weird = PathBuf::from("/tmp/mike's project");
    let err = render_cargo_toml(&weird).expect_err("should reject");
    assert!(err.contains("single quote"), "msg was: {err}");
}

/// Windows native builds produce `mlpl-build-user.exe`; the candidate list
/// previously hard-coded unsuffixed names, so mlpl-build reported "no
/// expected output found" after a successful build. Assert the suffix is
/// honored.
#[test]
fn candidate_names_appends_exe_suffix_on_windows() {
    let names = candidate_names(None, ".exe");
    assert_eq!(
        names,
        vec![
            "mlpl-build-user.exe".to_string(),
            "mlpl_build_user.exe".to_string()
        ]
    );
}

#[test]
fn candidate_names_unsuffixed_on_unix() {
    let names = candidate_names(None, "");
    assert_eq!(
        names,
        vec!["mlpl-build-user".to_string(), "mlpl_build_user".to_string()]
    );
}

#[test]
fn candidate_names_uses_wasm_extension_for_wasm32_target() {
    // Wasm extension wins over the host's exe suffix.
    let names = candidate_names(Some("wasm32-unknown-unknown"), ".exe");
    assert_eq!(
        names,
        vec![
            "mlpl-build-user.wasm".to_string(),
            "mlpl_build_user.wasm".to_string()
        ]
    );
}
