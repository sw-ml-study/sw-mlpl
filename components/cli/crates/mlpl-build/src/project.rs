//! The temp cargo project: locate the workspace, scaffold a tiny
//! project whose `main.rs` wraps the lowered program, compile it
//! with cargo, and copy out the produced binary.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::args::Args;
use crate::template;

/// Repo/workspace root, resolved from this crate's manifest dir at
/// build time (the generated project path-depends on `mlpl`).
pub(crate) fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// The include sandbox root: `source_dir` if given, else the input's
/// parent directory. A bare input path (`hello.mlpl`) has an EMPTY
/// parent (`Some("")`, not `None`), which is not a canonicalizable
/// directory, so it is dropped and the root falls through to ".".
pub(crate) fn resolve_root_dir(input: &Path, source_dir: Option<&Path>) -> PathBuf {
    let parent = input.parent().filter(|p| !p.as_os_str().is_empty());
    source_dir
        .or(parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Scaffold a temp cargo project (`Cargo.toml` + `src/`) that
/// path-depends on the workspace `mlpl` facade crate.
pub(crate) fn make_temp_project(workspace: &Path) -> Result<PathBuf, String> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("clock: {e}"))?
        .as_nanos();
    let tmp = base.join(format!("mlpl-build-{pid}-{nanos}"));
    std::fs::create_dir_all(tmp.join("src"))
        .map_err(|e| format!("creating temp dir {}: {e}", tmp.display()))?;
    let cargo_toml = template::render_cargo_toml(workspace)?;
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("writing Cargo.toml: {e}"))?;
    Ok(tmp)
}

/// Write the temp project's `main.rs`: the lowered program block as
/// the result, printed via the `CVal` `Display` (a scalar array shows
/// its value, a Result shows `ok(..)` / `err(..)`, matching the
/// interpreter) -- not `.arr()`, which would panic on a Result-valued
/// program (e.g. one ending in `write_stdout`). Every binding lowers
/// to `let mut` (so loop accumulators can rebind), which makes
/// single-assignment bindings `unused_mut` -- expected and harmless
/// in generated code, so the file allows it.
pub(crate) fn write_main_rs(tmp: &Path, lowered: &str) -> Result<(), String> {
    let main_rs = format!(
        "#![allow(unused_mut)]\n\
         fn main() {{\n\
             let result = {lowered};\n\
             println!(\"{{}}\", result);\n\
         }}\n"
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs).map_err(|e| format!("writing main.rs: {e}"))
}

/// Compile the temp project with cargo and copy the produced binary
/// to the requested output path.
pub(crate) fn build_and_emit(tmp: &Path, args: &Args) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--quiet"]).current_dir(tmp);
    if let Some(triple) = &args.target {
        cmd.args(["--target", triple]);
    }
    let status = cmd.status().map_err(|e| format!("invoking cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }
    // cargo places the binary at `<target>/[<triple>/]release/<name>`
    // where `<name>` is `mlpl-build-user` or `mlpl_build_user` (both
    // observed) with the platform executable suffix (`.wasm` for the
    // wasm32 target).
    let mut release_dir = tmp.join("target");
    if let Some(d) = args.target.as_deref() {
        release_dir.push(d);
    }
    release_dir.push("release");
    let candidates =
        template::candidate_names(args.target.as_deref(), std::env::consts::EXE_SUFFIX);
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
        .map(|_| ())
        .map_err(|e| format!("copying binary to {}: {e}", args.output.display()))
}
