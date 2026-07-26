//! Focused smoke test for the "Game of Life (APL classic)" demo:
//! every line lexes + parses + evals in one shared env, without
//! waiting on the full all-demos sweep.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};
use mlpl_web_demos::DEMOS;

#[test]
fn life_demo_evals() {
    let demo = DEMOS
        .iter()
        .find(|d| d.name == "Game of Life (APL classic)")
        .expect("life demo present in registry");
    let mut env = Environment::new();
    for (i, line) in demo.lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let toks = lex(line).unwrap_or_else(|e| panic!("[line {i}] lex: {e:?}"));
        let prog = parse(&toks).unwrap_or_else(|e| panic!("[line {i}] parse: {e:?}"));
        eval_program_value(&prog, &mut env).unwrap_or_else(|e| panic!("[line {i}] eval: {e:?}"));
    }
}
