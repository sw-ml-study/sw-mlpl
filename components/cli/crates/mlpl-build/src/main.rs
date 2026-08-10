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
use mlpl_lower_rs::LowerConfig;
use mlpl_parser_ast::Expr;
use quote::quote;

mod source_load;
mod template;

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
    // Resolve `include`s the same way the interpreter's script mode
    // does (source_load::load_stmts), then lower the flattened AST.
    let stmts = source_load::load_stmts(&args.input, args.source_dir.as_deref())?;
    let lowered = lower_program(&stmts)?;
    let tmp = make_temp_project(&workspace_root())?;
    write_main_rs(&tmp, &lowered)?;
    build_and_emit(&tmp, args)
}

/// Compile the generated temp project with cargo and copy the
/// produced binary to the requested output path.
fn build_and_emit(tmp: &Path, args: &Args) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--quiet"]).current_dir(tmp);
    if let Some(triple) = &args.target {
        cmd.args(["--target", triple]);
    }
    let status = cmd.status().map_err(|e| format!("invoking cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }

    // Locate the produced binary. cargo places it at
    // `<target-dir>/[<triple>/]release/<name>[<suffix>]`, where
    // `<name>` is either `mlpl-build-user` (dashes preserved) or
    // `mlpl_build_user` (dashes -> underscores; both variants
    // observed across cargo versions) and `<suffix>` is the
    // platform's executable suffix:
    //   - native build: `.exe` on Windows, empty elsewhere
    //     (via `std::env::consts::EXE_SUFFIX`)
    //   - wasm32 target: `.wasm`
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
        .map_err(|e| format!("copying binary to {}: {e}", args.output.display()))?;
    Ok(())
}

/// Lower the (include-expanded) program to a Rust block. Paths are
/// emitted through the `mlpl` facade's hidden `__rt` runtime alias
/// (the same path the `mlpl!` macro uses), so the generated temp
/// project needs only `mlpl` as a dependency. Lowering here surfaces
/// unsupported-construct / static-label errors with a stable
/// "mlpl-build: ..." prefix before we spin up cargo.
fn lower_program(stmts: &[Expr]) -> Result<String, String> {
    let cfg = LowerConfig {
        rt_path: quote! { ::mlpl::__rt },
    };
    let lowered =
        mlpl_lower_rs::lower_with_config(stmts, &cfg).map_err(|e| format!("lower error: {e}"))?;
    Ok(lowered.to_string())
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
    let cargo_toml = template::render_cargo_toml(workspace)?;
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("writing Cargo.toml: {e}"))?;
    Ok(tmp)
}

fn write_main_rs(tmp: &Path, lowered: &str) -> Result<(), String> {
    // `lowered` is already a Rust block expression (`{ ...; value }`)
    // emitted by mlpl-lower-rs, referencing `::mlpl::__rt::...`.
    let main_rs = format!(
        "fn main() {{\n\
             let result = {lowered};\n\
             println!(\"{{}}\", result.arr().data()[0]);\n\
         }}\n"
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs)
        .map_err(|e| format!("writing main.rs: {e}"))?;
    Ok(())
}
