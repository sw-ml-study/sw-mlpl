//! Emit `SERVE_BUILD_SHA` / `SERVE_BUILD_TIMESTAMP` (UTC Zulu) so
//! `/v1/health` can report the server's build and the web UI can
//! detect a UI-vs-server build skew. Pattern lifted from
//! apps/mlpl-web's build_env.rs.

use std::process::Command;

pub fn emit_build_env_vars() {
    let sha = run_cmd("git", &["rev-parse", "--short", "HEAD"]);
    let timestamp = run_cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    println!("cargo:rustc-env=SERVE_BUILD_SHA={sha}");
    println!("cargo:rustc-env=SERVE_BUILD_TIMESTAMP={timestamp}");
    // Best-effort: re-stamp when HEAD moves (a plain code rebuild
    // recompiles this crate and re-runs the script regardless).
    println!("cargo:rerun-if-changed=../../../../.git/HEAD");
}

fn run_cmd(prog: &str, args: &[&str]) -> String {
    Command::new(prog)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
