//! Version + build metadata for mlpl-repl.
//!
//! Compile-time env vars come from `build.rs`. Runtime callers get
//! either a rich `print()` for `--version` output or a multi-line
//! `banner()` for the REPL startup splash and the `:version` command.

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

const REPOSITORY: &str = "https://github.com/sw-ml-study/sw-mlpl";
const LICENSE: &str = "MIT OR Apache-2.0";
const COPYRIGHT: &str = "Copyright (c) 2026 Michael A Wright";

/// Print the rich version block to stdout. Used by the
/// `--version` / `-V` flag. Mirrors the Softwarewrighter CLI
/// convention (name+version, copyright, license, repository,
/// then a build-information block).
pub fn print() {
    println!(
        "{} {}\n{COPYRIGHT}\nLicense: {LICENSE}\nRepository: {REPOSITORY}\n\n\
         Build Information:\n  Host: {}\n  Commit: {}\n  Timestamp: {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_HOST"),
        env!("GIT_HASH"),
        env!("BUILD_TIMESTAMP"),
    );
}

/// The full `--help` block. Content lives in the sibling
/// `cli_help.txt` (per the "content in text files" convention)
/// and is embedded at compile time.
pub fn help_long() -> &'static str {
    include_str!("cli_help.txt")
}

/// The short `-h` usage synopsis. Brief by design; points at
/// `--help` for the full flag list. Following the sw-install
/// convention, `-h` is short and `--help` is the long form.
pub fn help_short() -> String {
    format!(
        "mlpl-repl {} -- MLPL interactive REPL and script runner\n\n\
         Usage:\n  \
         mlpl-repl                          Start the interactive REPL\n  \
         mlpl-repl <script.mlpl> [-- ARGS]  Run a script (args after --)\n\n\
         Flags: -h short help, --help full help, -V version, -v verbose, --trace\n",
        env!("CARGO_PKG_VERSION"),
    )
}
