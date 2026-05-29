//! Saga 75: render_cargo_toml + candidate_names extracted from
//! main.rs to keep its module function count under the limit.

use std::path::Path;

/// Build the temp project's Cargo.toml string. Extracted from
/// `make_temp_project` so the OS-path / TOML-escape interaction
/// (issue #3) can be unit-tested without spawning `cargo build`.
///
/// Issue #3 root cause: the previous version formatted the
/// `path = "..."` value as a TOML *basic* string with double
/// quotes. On Windows the workspace path contains backslashes
/// (`C:\Users\bill\...`) and TOML treats `\U` inside a basic
/// string as the start of an 8-digit-hex unicode escape, so
/// `cargo build` failed with "invalid unicode 8-digit hex code"
/// before ever invoking rustc.
///
/// Fix: build the dependency path with `PathBuf::join` so
/// separators are OS-native, and emit it as a TOML *literal*
/// string (single quotes). Literal strings forbid escape
/// processing, so backslashes survive verbatim. Forward slashes
/// also work (Cargo accepts both on Windows), but literal
/// strings handle the general case including UNC paths.
pub(crate) fn render_cargo_toml(workspace: &Path) -> Result<String, String> {
    let mlpl_crate = workspace.join("crates").join("mlpl");
    let mlpl_crate_str = mlpl_crate
        .to_str()
        .ok_or_else(|| format!("non-UTF-8 path: {}", mlpl_crate.display()))?;
    if mlpl_crate_str.contains('\'') {
        return Err(format!(
            "workspace path {mlpl_crate_str:?} contains a single quote, \
             which would break the TOML literal-string we emit"
        ));
    }
    Ok(format!(
        "[package]\n\
         name = \"mlpl-build-user\"\n\
         edition = \"2024\"\n\
         version = \"0.0.0\"\n\
         \n\
         [[bin]]\n\
         name = \"mlpl-build-user\"\n\
         path = \"src/main.rs\"\n\
         \n\
         [dependencies]\n\
         mlpl = {{ path = '{mlpl_crate_str}' }}\n"
    ))
}

/// Candidate filenames cargo might produce for the temp
/// project's bin target. Two variations:
/// 1. Cargo crates have historically used either dashed or
///    underscored binary names (depends on cargo version);
///    we try both.
/// 2. The file extension depends on `target`: `.wasm` for
///    `wasm32-unknown-unknown`, otherwise the platform's
///    executable suffix (`.exe` on Windows, empty on
///    Mac/Linux). Issue #3.5: Windows builds produced
///    `mlpl-build-user.exe` but the lookup only checked
///    unsuffixed names; cargo build "succeeded" but
///    mlpl-build still bailed.
pub(crate) fn candidate_names(target: Option<&str>, exe_suffix: &str) -> Vec<String> {
    let suffix = if target == Some("wasm32-unknown-unknown") {
        ".wasm"
    } else {
        exe_suffix
    };
    vec![
        format!("mlpl-build-user{suffix}"),
        format!("mlpl_build_user{suffix}"),
    ]
}
