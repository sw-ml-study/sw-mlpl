//! Saga 75: tests extracted from main.rs to keep main.rs
//! under the sw-checklist module-fn count limit. Included via
//! `#[path]` so the scope and `super::*` references still resolve
//! against `main.rs`.

let win_path = PathBuf::from(r"C:\Users\bill\Documents\Projects\sw-mlpl");
        let toml = crate::template::render_cargo_toml(&win_path).expect("render");
        // The whole path -- backslashes intact -- must be
        // inside single quotes (TOML literal string).
        assert!(
            toml.contains(r"path = 'C:\Users\bill\Documents\Projects\sw-mlpl"),
            "Windows backslashes were not preserved verbatim, got:\n{toml}"
        );
        // And it must NOT be inside a double-quoted basic
        // string -- that is what triggered issue #3.
        assert!(
            !toml.contains(r#"path = "C:\"#),
            "path was emitted as a basic string; TOML would \
             interpret backslashes as escapes. Full output:\n{toml}"
        );
        // Sanity: the `\U` in `\Users` is the specific sequence
        // that produced "invalid unicode 8-digit hex code" in
        // the issue. Confirm the line containing it is the
        // literal-string form.
        let line = toml
            .lines()
            .find(|l| l.contains(r"\Users"))
            .expect("path line");
        assert!(
            line.contains("'") && !line.contains('"'),
            "the path line still uses double quotes: {line:?}"
        );
