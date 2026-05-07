//! `mlpl-build` -- compile an MLPL source file to a native binary.
//!
//! Usage:
//!     mlpl-build <input.mlpl> -o <output> [--target <triple>]
//!
//! Implementation: read the source, lex + parse + lower eagerly so a
//! syntax or static label error fails before we spin up cargo. Then
//! generate a tiny temp cargo project whose `main.rs` wraps the
//! source in `mlpl::mlpl! { ... }` and prints the scalar result.
//! Shell out to `cargo build --release`, forwarding `--target` when
//! asked, and move the resulting binary to the requested output
//! path.
//!
//! The generated program depends on the workspace `mlpl` facade
//! crate via a path dependency that resolves back to this workspace
//! (located at build time via `CARGO_MANIFEST_DIR`). When `mlpl`
//! ships to crates.io this becomes a version dep; keeping it path-
//! based today keeps the dev story self-contained.

mod args;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use args::{Args, USAGE};

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match Args::parse(&argv) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("mlpl-build: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    let src = std::fs::read_to_string(&args.input)
        .map_err(|e| format!("reading {}: {e}", args.input.display()))?;
    eager_parse_check(&src, &args.input)?;

    let workspace = workspace_root();
    let tmp = make_temp_project(&workspace)?;
    write_main_rs(&tmp, &src)?;

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--quiet"])
        .current_dir(&tmp);
    if let Some(triple) = &args.target {
        cmd.args(["--target", triple]);
    }
    let status = cmd.status().map_err(|e| format!("invoking cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }

    // Locate the produced binary. cargo places it at
    // `<target-dir>/[<triple>/]release/<name>[.wasm]`, where `<name>`
    // is either `mlpl-build-user` (native, dashes preserved) or
    // `mlpl_build_user.wasm` (wasm, dashes -> underscores). Both
    // variants have been observed across cargo versions; try each.
    let mut release_dir = tmp.join("target");
    if let Some(d) = args.target.as_deref() {
        release_dir.push(d);
    }
    release_dir.push("release");
    let candidates: &[&str] = if args.target.as_deref() == Some("wasm32-unknown-unknown") {
        &["mlpl_build_user.wasm", "mlpl-build-user.wasm"]
    } else {
        &["mlpl-build-user", "mlpl_build_user"]
    };
    let binary = candidates
        .iter()
        .map(|name| release_dir.join(name))
        .find(|p| p.exists())
        .ok_or_else(|| {
            format!(
                "cargo build reported success but no expected output found in {}",
                release_dir.display()
            )
        })?;
    std::fs::copy(&binary, &args.output)
        .map_err(|e| format!("copying binary to {}: {e}", args.output.display()))?;
    Ok(())
}

fn eager_parse_check(src: &str, input: &Path) -> Result<(), String> {
    let tokens =
        mlpl_parser::lex(src).map_err(|e| format!("lex error in {}: {e}", input.display()))?;
    let stmts = mlpl_parser::parse(&tokens)
        .map_err(|e| format!("parse error in {}: {e}", input.display()))?;
    // Lowering eagerly surfaces static label mismatches with a
    // stable "mlpl-build: ..." prefix instead of cascading into a
    // confusing rustc error inside the temp cargo project.
    mlpl_lower_rs::lower(&stmts).map_err(|e| format!("lower error in {}: {e}", input.display()))?;
    Ok(())
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

fn make_temp_project(workspace: &Path) -> Result<PathBuf, String> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("clock: {e}"))?
        .as_nanos();
    let tmp = base.join(format!("mlpl-build-{pid}-{nanos}"));
    std::fs::create_dir_all(tmp.join("src"))
        .map_err(|e| format!("creating temp dir {}: {e}", tmp.display()))?;
    let cargo_toml = render_cargo_toml(workspace)?;
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("writing Cargo.toml: {e}"))?;
    Ok(tmp)
}

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
fn render_cargo_toml(workspace: &Path) -> Result<String, String> {
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

fn write_main_rs(tmp: &Path, src: &str) -> Result<(), String> {
    // MLPL source with newlines goes inside the `mlpl!` macro; the
    // macro rewrites newlines to spaces (see mlpl-macro/src/lib.rs),
    // but statement separation in the macro requires `;`, so swap
    // REPL-style newlines for semicolons here.
    let macro_body: String = src
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("; ");
    let main_rs = format!(
        "use mlpl::mlpl;\n\
         fn main() {{\n\
             let result = mlpl! {{ {macro_body} }};\n\
             println!(\"{{}}\", result.data()[0]);\n\
         }}\n"
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs)
        .map_err(|e| format!("writing main.rs: {e}"))?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #3 regression: a Windows-style workspace path
    /// containing backslashes (and a `\U` that TOML's
    /// basic-string parser would interpret as a unicode escape)
    /// must be emitted inside a TOML *literal* string so the
    /// backslashes survive verbatim and never reach the basic-
    /// string escape parser. The separator after `crates` /
    /// `mlpl` may be either `/` or `\` depending on host OS
    /// (PathBuf::join uses native), and Cargo accepts both on
    /// Windows -- what matters is single quotes, not slashes.
    #[test]
    fn render_cargo_toml_handles_windows_backslash_path() {
        let win_path = PathBuf::from(r"C:\Users\bill\Documents\Projects\sw-mlpl");
        let toml = render_cargo_toml(&win_path).expect("render");
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
}
