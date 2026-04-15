//! CLI argument parsing for `mlpl-build`.

use std::path::PathBuf;

pub const USAGE: &str = "usage: mlpl-build <input.mlpl> -o <output> [--target <triple>]";

#[derive(Debug)]
pub struct Args {
    pub input: PathBuf,
    pub output: PathBuf,
    pub target: Option<String>,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Self, String> {
        let mut input: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut target: Option<String> = None;
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "-o" => {
                    i += 1;
                    output = Some(
                        argv.get(i)
                            .ok_or_else(|| "missing -o argument".to_string())?
                            .into(),
                    );
                }
                "--target" => {
                    i += 1;
                    target = Some(
                        argv.get(i)
                            .ok_or_else(|| "missing --target argument".to_string())?
                            .clone(),
                    );
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
        })
    }
}
