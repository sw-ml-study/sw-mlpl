//! Saga 82: web demo registry smoke -- walks every entry in
//! `DEMOS`, lexes + parses + evals each line in a fresh shared
//! environment to catch syntax / runtime drift the moment a
//! demo string in this file falls behind language changes.
//! Mirrors what the "Run demo" button does in the browser,
//! minus the visualization output.
//!
//! Skipped: REPL slash-commands (`:tags x`) and pure
//! comments are not parseable as expressions; the browser
//! handles those as side-channel commands. For long-running
//! training demos we follow the same split as
//! `all_demos_smoke`: a quick test exercises everything
//! except the heavy ones, and a `#[ignore]`-gated test
//! covers the heavies on demand.
//!
//! `PROGRESS_NOTES` invariant: every entry's `demo` matches
//! a real `Demo::name` and `line_idx` is within that demo's
//! `lines` length. A mismatch would silently fail to render
//! the heads-up note in the browser.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};
use mlpl_web_demos::{DEMOS, PROGRESS_NOTES};
use std::collections::HashSet;

/// Demos that call external services or do heavy training.
const SKIP_DEMOS: &[&str] = &[
    "LLM Tool Use",
    "MLX Remote Runner",
    "Tiny LM",
    "Tiny LM Generate",
    "Moons MLP",
    "Circles MLP",
    "Transformer Block",
    "Pets: cat vs dog (quick)",
    "Pets: predict + gallery",
    "Pets: multi-head ViT (quick + viz)",
];

fn run_demo(demo_name: &str, lines: &[&str]) -> Result<(), String> {
    let mut env = Environment::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with('#')
            || (trimmed.starts_with(':') && !trimmed.starts_with("::"))
        {
            continue;
        }
        let toks = lex(line).map_err(|e| format!("[{demo_name} line {i}] lex: {e:?}"))?;
        let prog = parse(&toks).map_err(|e| format!("[{demo_name} line {i}] parse: {e:?}"))?;
        eval_program_value(&prog, &mut env)
            .map_err(|e| format!("[{demo_name} line {i}] eval: {e:?}"))?;
    }
    Ok(())
}

#[test]
fn every_quick_web_demo_runs() {
    let mut failures: Vec<String> = Vec::new();
    for demo in DEMOS.iter() {
        if SKIP_DEMOS.contains(&demo.name) {
            continue;
        }
        if let Err(msg) = run_demo(demo.name, demo.lines) {
            failures.push(msg);
        }
    }
    assert!(
        failures.is_empty(),
        "{} web demo(s) regressed:\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
}

#[test]
#[ignore = "heavy training demos take 30+s; run with --ignored"]
fn every_heavy_web_demo_runs() {
    let mut failures: Vec<String> = Vec::new();
    for demo in DEMOS.iter() {
        if !SKIP_DEMOS.contains(&demo.name) {
            continue;
        }
        if matches!(demo.name, "LLM Tool Use" | "MLX Remote Runner") {
            continue;
        }
        if let Err(msg) = run_demo(demo.name, demo.lines) {
            failures.push(msg);
        }
    }
    assert!(
        failures.is_empty(),
        "{} heavy web demo(s) regressed:\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
}

#[test]
fn progress_notes_reference_real_demo_lines() {
    let demos: HashSet<&str> = DEMOS.iter().map(|d| d.name).collect();
    let mut bad: Vec<String> = Vec::new();
    for note in PROGRESS_NOTES.iter() {
        if !demos.contains(note.demo) {
            bad.push(format!("unknown demo {:?}", note.demo));
            continue;
        }
        let demo = DEMOS.iter().find(|d| d.name == note.demo).unwrap();
        if note.line_idx >= demo.lines.len() {
            bad.push(format!(
                "{}: line_idx {} >= lines.len() {}",
                note.demo,
                note.line_idx,
                demo.lines.len()
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "PROGRESS_NOTES drift:\n  - {}",
        bad.join("\n  - ")
    );
}
