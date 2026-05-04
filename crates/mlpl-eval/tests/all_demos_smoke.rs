//! Demo-coverage smoke test.
//!
//! `every_quick_demo_runs` walks every fast `demos/*.mlpl` file
//! and asserts it lexes, parses, and evaluates without panic or
//! `EvalError`. Heavy training-loop demos (`tiny_lm.mlpl`,
//! `tiny_lm_generate.mlpl`, `transformer_block.mlpl`,
//! `moons_mlp.mlpl`, `circles_mlp.mlpl`) live in
//! `every_heavy_demo_runs` which is `#[ignore]`'d so the default
//! `cargo test` finishes in seconds. Run the full sweep with
//! `cargo test --test all_demos_smoke -- --ignored`.
//!
//! Demos already covered by a dedicated integration test
//! (attention, kmeans, pca, embedding_viz,
//! lora_finetune[_mlx], neural_thicket[_mlx], tiny_lm_mlx) are
//! skipped here to avoid duplicate coverage. External-service
//! demos (`llm_tool.mlpl` -> live Ollama, `mlx_remote.mlpl` ->
//! peer service) are also skipped.
//!
//! Failure reporting collects every demo's first error so one
//! bad refactor surfaces the full set of broken demos in one
//! shot rather than failing fast on the first.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};
use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DEDICATED: &[&str] = &[
    "attention.mlpl",
    "embedding_viz.mlpl",
    "kmeans.mlpl",
    "lora_finetune.mlpl",
    "lora_finetune_mlx.mlpl",
    "neural_thicket.mlpl",
    "neural_thicket_mlx.mlpl",
    "pca.mlpl",
    "tiny_lm_mlx.mlpl",
];

const SKIP_EXTERNAL: &[&str] = &["llm_tool.mlpl", "mlx_remote.mlpl"];

/// Demos with substantial training loops (hundreds of steps,
/// tokenizer training, full-vocab BPE). Covered separately by
/// `every_heavy_demo_runs` (`#[ignore]`'d) so the default smoke
/// pass stays fast.
const HEAVY_TRAINING: &[&str] = &[
    "circles_mlp.mlpl",
    "moons_mlp.mlpl",
    "tiny_lm.mlpl",
    "tiny_lm_generate.mlpl",
    "transformer_block.mlpl",
];

fn discover() -> Vec<PathBuf> {
    let demos_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../demos");
    let entries = fs::read_dir(&demos_dir)
        .unwrap_or_else(|e| panic!("read demos dir {}: {e}", demos_dir.display()));
    let mut out: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mlpl"))
        .collect();
    out.sort();
    out
}

fn name_of(path: &Path) -> &str {
    path.file_name().and_then(|s| s.to_str()).unwrap_or("?")
}

fn run_one(path: &Path) -> Result<(), String> {
    let src = fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    let tokens = lex(&src).map_err(|e| format!("lex error: {e}"))?;
    let stmts = parse(&tokens).map_err(|e| format!("parse error: {e}"))?;
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env).map_err(|e| {
        let msg = format!("eval error: {e}");
        if msg.len() > 200 {
            format!("{}...", &msg[..200])
        } else {
            msg
        }
    })?;
    Ok(())
}

fn run_filter(want_heavy: bool) {
    let demos = discover();
    assert!(
        !demos.is_empty(),
        "no demos discovered -- the test runner cannot find demos/"
    );
    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0;
    for path in &demos {
        let name = name_of(path);
        if SKIP_DEDICATED.contains(&name) || SKIP_EXTERNAL.contains(&name) {
            continue;
        }
        let is_heavy = HEAVY_TRAINING.contains(&name);
        if is_heavy != want_heavy {
            continue;
        }
        ran += 1;
        if let Err(msg) = run_one(path) {
            failures.push(format!("{name}: {msg}"));
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} demos ran, {} failed:\n  - {}",
            ran,
            failures.len(),
            failures.join("\n  - ")
        );
    }
}

#[test]
fn every_quick_demo_runs() {
    run_filter(false);
}

#[test]
#[ignore = "heavy training loops; run with --ignored or in CI"]
fn every_heavy_demo_runs() {
    run_filter(true);
}
