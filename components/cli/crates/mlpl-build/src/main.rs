//! `mlpl-build` -- compile an MLPL source file to a native binary.
//!
//! Usage:
//!     mlpl-build <input.mlpl> -o <output> [--target <triple>] [--source-dir <dir>]
//!
//! Implementation: resolve `include`s (like the interpreter's script
//! mode), lower the flattened AST to Rust, scaffold a tiny temp cargo
//! project whose `main.rs` runs the lowered program, `cargo build
//! --release` it, and copy the binary to the requested path. The
//! temp-project scaffold/build/emit lives in `project`; include
//! resolution in `source_load`; CLI parsing in `args`; the Cargo.toml
//! + binary-name templates in `template`.

mod args;
mod project;
mod source_load;
mod template;

use std::process::ExitCode;

use args::{Args, USAGE};
use mlpl_lower_rs::LowerConfig;
use mlpl_parser_ast::Expr;
use quote::quote;

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
    let tmp = project::make_temp_project(&project::workspace_root())?;
    project::write_main_rs(&tmp, &lowered)?;
    project::build_and_emit(&tmp, args)
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
