//! Focused smoke test for the "Life Pattern Zoo (20x20)" demo.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};
use mlpl_web_demos::DEMOS;

#[test]
fn life_zoo_demo_evals() {
    let demo = DEMOS
        .iter()
        .find(|d| d.name == "Life Pattern Zoo (20x20)")
        .expect("zoo demo present in registry");
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
