//! Build script that emits BUILD_HOST, GIT_HASH, and BUILD_TIMESTAMP
//! environment variables for the binary to read at runtime via env!.
//!
//! Pattern lifted from sw-cli-tools/sw-cli-gen so every Software-
//! wrighter CLI tool reports version info the same way.

use std::process::Command;

fn main() {
    let host = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_HOST={host}");

    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=GIT_HASH={hash}");

    let timestamp = chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%z")
        .to_string();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={timestamp}");

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}
