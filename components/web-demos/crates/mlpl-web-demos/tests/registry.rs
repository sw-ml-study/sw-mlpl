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
    // 1500-step fine-tune over the full policy dataset: ~minutes on the
    // CPU fallback (it runs in ~2s on the GPU). Heavy test covers it.
    "MLX tic-tac-toe fine-tune",
    // 6000-step board-policy fine-tune + self-play minimax: ~minutes on
    // the CPU fallback (it runs in ~10s on the GPU). Heavy test covers it.
    "CUDA tic-tac-toe fine-tune",
    "Tiny LM",
    "Tiny LM Generate",
    // 200-step train over a multi-timestep noised dataset: heavy. The
    // diffusion_2d eval test covers correctness; heavy test covers the run.
    "Diffusion (2D points)",
    // 1200-step adversarial train (D + 2x G per step): heavy.
    "GAN (2D circle)",
    "Moons MLP",
    // ~150 Adam steps over six chunks plus per-chunk train+val scoring on
    // a 200-point set: heavy on the dev profile. The train_val_curve eval
    // test covers the dispatch + accumulation; heavy test covers the run.
    "Watch a Model Learn (overfitting)",
    // Companion: 100 Adam steps on a 200-point train + 200-point val set.
    // Heavy on dev; generalize_demo_tests.rs covers the run + accuracy.
    "Watch a Model Generalize (no overfitting)",
    // 150 Adam steps over a hand-rolled net + L2 penalty. Heavy on dev;
    // regularization_lr_tests.rs covers the run + accuracy.
    "Taming Overfitting: Weight Decay",
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
fn learning_rate_demo_evals() {
    // Light (scalar GD), so eval the exact generated lines here rather than
    // waiting for the full quick smoke -- catches any demos.toml typo.
    let name = "Learning Rate: too big, too small, just right";
    let d = DEMOS
        .iter()
        .find(|d| d.name == name)
        .expect("LR demo present");
    run_demo(d.name, d.lines).expect("LR demo evals");
}

#[test]
fn visualizations_demo_evals() {
    // The viz tour renders 12 different plots from toy data -- the synthetic
    // shapes are the risk, so eval the exact generated lines here.
    let d = DEMOS
        .iter()
        .find(|d| d.name == "Visualizations")
        .expect("Visualizations demo present");
    run_demo(d.name, d.lines).expect("Visualizations demo evals");
}

#[test]
fn gradient_flow_demo_evals() {
    // Light (one forward + four grads, no training loop): eval the exact
    // generated lines here rather than waiting for the full quick smoke.
    let name = "Gradient Flow (backprop)";
    let d = DEMOS
        .iter()
        .find(|d| d.name == name)
        .expect("gradient-flow demo present");
    run_demo(d.name, d.lines).expect("gradient-flow demo evals");
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
