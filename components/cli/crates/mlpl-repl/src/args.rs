//! CLI argument parsing for `mlpl-repl`. `parse` turns raw argv
//! into a `Config` that the run phase consumes, so `main` does no
//! arg parsing of its own.

use std::path::PathBuf;

use crate::script_mode;

/// Info-only flags that print and exit before any work. `-V` /
/// `--version` win over help; `--help` is the long form, `-h` the
/// short usage (the sw-install convention).
pub(crate) enum Info {
    None,
    Version,
    HelpLong,
    HelpShort,
}

/// Everything the run phase needs, parsed once from argv.
pub(crate) struct Config {
    pub info: Info,
    /// Pre-`--` args, kept for the connect-subcommand dispatch.
    pub connect_args: Vec<String>,
    /// Resolved script path; `None` selects the interactive REPL.
    pub script: Option<String>,
    /// Post-`--` args, exposed to the script via `args()`.
    pub script_args: Vec<String>,
    pub svg_out: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub source_dir: Option<PathBuf>,
    pub exp_dir: Option<PathBuf>,
    pub trace: bool,
    pub verbose: bool,
    /// `--babel-session`: persistent stdin block loop for ob-mlpl.
    pub babel_session: bool,
}

/// Parse raw argv (including argv[0]) into a `Config`.
pub(crate) fn parse(raw: Vec<String>) -> Config {
    let (args, script_args) = split_separator(&raw);
    let flag = |name: &str| flag_value(&args, name);
    Config {
        info: detect_info(&args),
        script: script_mode::resolve_script_path(&args, &flag),
        svg_out: flag("--svg-out").map(PathBuf::from),
        data_dir: flag("--data-dir").map(PathBuf::from),
        source_dir: flag("--source-dir").map(PathBuf::from),
        exp_dir: flag("--exp-dir").map(PathBuf::from),
        trace: args.iter().any(|a| a == "--trace"),
        // Quiet by default; -v/--verbose echoes each script line.
        verbose: args.iter().any(|a| a == "-v" || a == "--verbose"),
        babel_session: args.iter().any(|a| a == "--babel-session"),
        script_args,
        connect_args: args,
    }
}

/// Split argv on `--`: args before go to mlpl-repl's flag parser,
/// args after become the script's `args()` (separator dropped).
/// Saga 31 step 003.
fn split_separator(raw: &[String]) -> (Vec<String>, Vec<String>) {
    match raw.iter().position(|a| a == "--") {
        Some(i) => (raw[..i].to_vec(), raw[i + 1..].to_vec()),
        None => (raw.to_vec(), Vec::new()),
    }
}

/// Which info-only flag (if any) was requested.
fn detect_info(args: &[String]) -> Info {
    if args.iter().any(|a| a == "-V" || a == "--version") {
        Info::Version
    } else if args.iter().any(|a| a == "--help") {
        Info::HelpLong
    } else if args.iter().any(|a| a == "-h") {
        Info::HelpShort
    } else {
        Info::None
    }
}

/// The value following `name` in `args`, if present.
fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|p| args.get(p + 1))
        .cloned()
}
