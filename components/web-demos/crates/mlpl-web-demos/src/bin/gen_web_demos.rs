//! Generate one runnable `.mlpl` file per web-playground demo, under
//! `work/reg-rs/web-demos/<slug>.mlpl`, from the canonical `DEMOS`
//! registry (which `build.rs` compiles from `demos.toml`). Each file
//! opens with a doc header -- the demo's name and intro prose -- so a
//! reader who opens the extracted file, not just the playground, sees
//! what it does; the header survives regeneration because it is
//! emitted here, not hand-added. The MLPL lines themselves are copied
//! verbatim, so the reg-rs regression baselines (which compare program
//! OUTPUT) are unaffected by the comment header.
//!
//! Usage: `cargo run -p mlpl-web-demos --bin gen-web-demos [OUT_DIR]`
//! (default OUT_DIR: `work/reg-rs/web-demos`).

use std::fs;
use std::path::Path;

use mlpl_web_demos::{capability_for, Demo, Device, DEMOS};

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "work/reg-rs/web-demos".to_string());
    let dir = Path::new(&out_dir);
    fs::create_dir_all(dir).expect("create out dir");
    let mut n = 0;
    for demo in DEMOS.iter().filter(|d| cpu_runnable(d.name)) {
        let path = dir.join(format!("{}.mlpl", slug(demo.name)));
        fs::write(&path, file_body(demo)).expect("write demo file");
        println!("wrote {}", path.display());
        n += 1;
    }
    println!("\n{n} CPU-runnable demos written to {out_dir}");
}

/// Only CPU demos runnable by a plain `mlpl-repl -f` are extracted:
/// connect-only / GPU (MLX, CUDA) / Ollama demos need a server or
/// device and would fail the reg-rs standalone run.
fn cpu_runnable(name: &str) -> bool {
    let cap = capability_for(name);
    !cap.requires_connect && cap.device == Device::Cpu
}

/// Derive a filesystem-safe slug from a demo name: lowercase, then
/// every run of non-`[a-z0-9]` characters becomes a single dash, with
/// leading/trailing dashes trimmed (so `Structure Zoo (rank / shape)`
/// -> `structure-zoo-rank-shape`, never a stray path separator).
fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_dash = true; // trims leading dashes
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// The full file text: doc header comment, a blank line, then the
/// demo's MLPL lines (each on its own line, trailing newline).
fn file_body(demo: &Demo) -> String {
    let mut out = header(demo);
    out.push('\n');
    for line in demo.lines.iter() {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The `#` doc header: the demo name, its intro prose (one `#` line
/// per intro line), and a provenance note pointing back at the source.
fn header(demo: &Demo) -> String {
    let mut out = format!("# {}\n#\n", demo.name);
    for line in demo.intro.lines() {
        out.push_str(&format!("# {line}\n"));
    }
    out.push_str(
        "#\n# Generated from demos.toml (the mlpl-web-demos DEMOS registry) by\n\
         # `cargo run -p mlpl-web-demos --bin gen-web-demos`. Do not edit here;\n\
         # edit demos.toml and regenerate.\n",
    );
    out
}
