//! Version + build metadata for mlpl-repl.
//!
//! Compile-time env vars come from `build.rs`. Runtime callers get
//! either a one-line `print()` for `--version` output or a multi-line
//! `banner()` for the REPL startup splash and the `:version` command.

/// One-line version string suitable for `mlpl-repl --version` output,
/// matching the convention used by other Softwarewrighter CLI tools.
#[must_use]
pub fn one_line() -> String {
    format!(
        "{} {} ({} {} {})",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_HOST"),
        env!("GIT_HASH"),
        env!("BUILD_TIMESTAMP"),
    )
}

/// Multi-line banner used both as the REPL startup splash and as the
/// output of the `:version` command. Includes the dot version, build
/// host, git short SHA, and ISO build timestamp.
#[must_use]
pub fn banner() -> String {
    format!(
        "MLPL v{} -- Array Programming Language for ML\n\
         build: host={} commit={} built={}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_HOST"),
        env!("GIT_HASH"),
        env!("BUILD_TIMESTAMP"),
    )
}

/// Print the one-line version string to stdout. Used by the
/// `--version` / `-V` command-line flag.
pub fn print() {
    println!("{}", one_line());
}
