//! Focused smoke test for the "Error Handling (two planes, two
//! bridges)" demo: every line lexes + parses + evals in one shared
//! env; lines whose comment declares an INTENTIONAL ERROR must fail
//! (the browser prints the error entry and continues).

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};
use mlpl_web_demos::DEMOS;

#[test]
fn error_handling_demo_evals() {
    let demo = DEMOS
        .iter()
        .find(|d| d.name == "Error Handling (two planes, two bridges)")
        .expect("error handling demo present in registry");
    let mut env = Environment::new();
    for (i, line) in demo.lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let toks = lex(line).unwrap_or_else(|e| panic!("[line {i}] lex: {e:?}"));
        let prog = parse(&toks).unwrap_or_else(|e| panic!("[line {i}] parse: {e:?}"));
        let result = eval_program_value(&prog, &mut env);
        if line.contains("INTENTIONAL ERROR") {
            assert!(result.is_err(), "[line {i}] expected the teaching error");
            continue;
        }
        result.unwrap_or_else(|e| panic!("[line {i}] eval: {e:?}"));
    }
}
