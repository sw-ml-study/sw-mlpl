//! Saga 33 step 007: emit BUILD_HOST / BUILD_SHA /
//! BUILD_TIMESTAMP env vars so the footer in `src/components.rs`
//! can render `host -- sha -- built` next to the legal blurb.
//! Extracted from `build.rs` so the entrypoint stays a thin
//! delegate; pattern lifted verbatim from
//! `~/github/sw-embed/web-sw-cor24-apl/build.rs`.

use std::process::Command;

pub fn emit_build_env_vars() {
    let host = run_cmd("hostname", &[]);
    let sha = run_cmd("git", &["rev-parse", "--short", "HEAD"]);
    let timestamp = run_cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    println!("cargo:rustc-env=BUILD_HOST={host}");
    println!("cargo:rustc-env=BUILD_SHA={sha}");
    println!("cargo:rustc-env=BUILD_TIMESTAMP={timestamp}");
    println!("cargo:rerun-if-changed=.git/HEAD");
}

fn run_cmd(prog: &str, args: &[&str]) -> String {
    Command::new(prog)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
