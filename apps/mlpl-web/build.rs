//! Emit BUILD_HOST, BUILD_SHA, BUILD_TIMESTAMP env vars so the
//! footer in `src/components.rs` can render `host -- sha --
//! built` next to the legal blurb. Pattern lifted verbatim from
//! `~/github/sw-embed/web-sw-cor24-apl/build.rs` so every
//! Software Wrighter live demo's footer renders the same way.
//!
//! `mlpl-repl` already has its own copy of this script for the
//! CLI banner; both crates avoid a shared build-helper crate
//! because the script is six lines of Command::new and not
//! worth a new path dep.

use std::process::Command;

fn main() {
    let host = Command::new("hostname")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let timestamp = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());

    println!("cargo:rustc-env=BUILD_HOST={host}");
    println!("cargo:rustc-env=BUILD_SHA={sha}");
    println!("cargo:rustc-env=BUILD_TIMESTAMP={timestamp}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}
