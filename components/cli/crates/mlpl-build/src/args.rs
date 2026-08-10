//! CLI argument parsing for `mlpl-build`.

use std::path::PathBuf;

pub const USAGE: &str =
    "usage: mlpl-build <input.mlpl> -o <output> [--target <triple>] [--source-dir <dir>]";

#[derive(Debug)]
pub struct Args {
    pub input: PathBuf,
    pub output: PathBuf,
    pub target: Option<String>,
    /// Include sandbox root (mirrors `mlpl-repl -f --source-dir`).
    /// Defaults to the input file's own directory.
    pub source_dir: Option<PathBuf>,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Self, String> {
        let mut input: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut target: Option<String> = None;
        let mut source_dir: Option<PathBuf> = None;
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "-o" => output = Some(take_value(argv, &mut i, "-o")?.into()),
                "--target" => target = Some(take_value(argv, &mut i, "--target")?.to_string()),
                "--source-dir" => {
                    source_dir = Some(take_value(argv, &mut i, "--source-dir")?.into())
                }
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                arg if arg.starts_with('-') => {
                    return Err(format!("unknown flag: {arg}"));
                }
                _ => {
                    if input.is_some() {
                        return Err(format!("unexpected positional argument: {}", argv[i]));
                    }
                    input = Some((&argv[i]).into());
                }
            }
            i += 1;
        }
        Ok(Self {
            input: input.ok_or_else(|| "missing <input.mlpl>".to_string())?,
            output: output.ok_or_else(|| "missing -o <output>".to_string())?,
            target,
            source_dir,
        })
    }
}

/// Advance past `flag` and return its value argument, erroring when
/// the flag is the last token.
fn take_value<'a>(argv: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    argv.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {flag} argument"))
}
