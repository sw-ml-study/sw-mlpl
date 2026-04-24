//! Saga 22 step 002: `calibrate_device` +
//! `estimate_hypothetical` + `feasible` builtins.
//!
//! Feasibility toolkit built on top of `estimate_train`:
//! - `calibrate_device()` -> gflops: benchmarks a
//!   matmul on the active device and caches the
//!   result in `env.set_string("mlpl_device_
//!   throughput_gflops", ...)` so subsequent
//!   `estimate_train` reads honest numbers.
//! - `estimate_hypothetical(name, steps, batch, seq
//!   [, dtype_bytes, lora_rank])` -> [5]: direct
//!   estimate for a hardcoded HF-scale model table
//!   (SmolLM-135M/360M/1.7B, Llama-3.2-1B,
//!   Qwen-2.5-0.5B) without needing to materialize
//!   weights in env.
//! - `feasible(est, [vram_budget, disk_budget,
//!   wall_budget])` -> 0/1: guard pattern; zeros in
//!   the budget slots mean "skip this dimension".

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(env: &mut Environment, src: &str) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

// ---- calibrate_device ----

#[test]
fn calibrate_device_returns_positive_gflops_and_caches_key() {
    let mut env = Environment::new();
    // Small size keeps the test fast; the inner
    // benchmark accepts an optional size override.
    let g = run(&mut env, "calibrate_device(128)");
    assert_eq!(g.rank(), 0, "gflops should be a scalar");
    let v = g.data()[0];
    assert!(v > 0.0 && v.is_finite(), "gflops must be positive, got {v}");
    let cached = env
        .get_string("mlpl_device_throughput_gflops")
        .expect("calibrate_device should write the env key");
    let parsed: f64 = cached.parse().expect("env key must be a float string");
    assert!(
        (parsed - v).abs() < 1e-6,
        "env key {parsed} should match return {v}"
    );
}

#[test]
#[ignore = "slow: 1024x1024 matmul x 10 takes several minutes on CPU; run via `cargo test -- --ignored`"]
fn calibrate_device_default_size_is_1024() {
    // Zero-arg form uses the documented 1024 default.
    // We don't assert the specific GFLOPS number (it
    // varies 100x across hardware) -- just that the
    // call succeeds and returns something positive.
    let mut env = Environment::new();
    let g = run(&mut env, "calibrate_device()");
    let v = g.data()[0];
    assert!(v > 0.0 && v.is_finite(), "gflops must be positive, got {v}");
}

// ---- estimate_hypothetical ----

#[test]
fn estimate_hypothetical_smollm_135m_param_count_in_range() {
    let mut env = Environment::new();
    let est = run(
        &mut env,
        "estimate_hypothetical(\"smollm-135m\", 100, 4, 512)",
    );
    assert_eq!(est.shape().dims(), &[5]);
    let params = est.data()[0];
    // SmolLM-135M is in the 130M-150M range; our
    // structural approximation should land within
    // 30%.
    assert!(
        (100e6..=200e6).contains(&params),
        "smollm-135m params {params} should be in the ~135M range"
    );
}

#[test]
fn estimate_hypothetical_scales_with_size() {
    let mut env = Environment::new();
    let s135 = run(
        &mut env,
        "estimate_hypothetical(\"smollm-135m\", 100, 4, 512)",
    );
    let s1b7 = run(
        &mut env,
        "estimate_hypothetical(\"smollm-1.7b\", 100, 4, 512)",
    );
    // Bigger model, more params (by at least 5x --
    // 135M -> 1.7B is nominally 12x).
    assert!(
        s1b7.data()[0] > 5.0 * s135.data()[0],
        "1.7B params {} should dwarf 135M params {}",
        s1b7.data()[0],
        s135.data()[0]
    );
    // More FLOPS too.
    assert!(s1b7.data()[3] > 5.0 * s135.data()[3]);
}

#[test]
fn estimate_hypothetical_lora_reduces_trainable_memory() {
    let mut env = Environment::new();
    let full = run(
        &mut env,
        "estimate_hypothetical(\"smollm-135m\", 100, 4, 512)",
    );
    let lora = run(
        &mut env,
        "estimate_hypothetical(\"smollm-135m\", 100, 4, 512, 8, 8)",
    );
    // LoRA at rank 8: disk unchanged (still saves
    // full base), params unchanged, but VRAM drops
    // because grad + Adam moments only cover the
    // tiny adapter set.
    assert_eq!(full.data()[0], lora.data()[0], "param count should match");
    assert_eq!(full.data()[2], lora.data()[2], "disk should match");
    assert!(
        lora.data()[1] < full.data()[1],
        "LoRA VRAM {} must be less than full-fine-tune VRAM {}",
        lora.data()[1],
        full.data()[1]
    );
}

#[test]
fn estimate_hypothetical_unknown_name_errors() {
    let mut env = Environment::new();
    let stmts =
        parse(&lex("estimate_hypothetical(\"made-up-model\", 10, 4, 64)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("unknown name should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_hypothetical") && msg.contains("made-up-model"),
        "got: {msg}"
    );
}

#[test]
fn estimate_hypothetical_wrong_arity_errors() {
    let mut env = Environment::new();
    let stmts = parse(&lex("estimate_hypothetical(\"smollm-135m\")").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_hypothetical") || msg.contains("arity"),
        "got: {msg}"
    );
}

// ---- feasible ----

#[test]
fn feasible_passes_when_all_budgets_met() {
    let mut env = Environment::new();
    env.set(
        "est".into(),
        arr(vec![5], vec![1000.0, 4096.0, 128.0, 1e6, 1.0]),
    );
    env.set("budget".into(), arr(vec![3], vec![1e9, 1e9, 60.0]));
    let ok = run(&mut env, "feasible(est, budget)");
    assert_eq!(ok.rank(), 0);
    assert_eq!(ok.data()[0], 1.0, "all budgets satisfied should be 1.0");
}

#[test]
fn feasible_fails_when_vram_exceeded() {
    let mut env = Environment::new();
    // vram = 5GB exceeds a 1GB budget
    env.set(
        "est".into(),
        arr(vec![5], vec![1000.0, 5e9, 128.0, 1e6, 1.0]),
    );
    env.set("budget".into(), arr(vec![3], vec![1e9, 1e12, 3600.0]));
    let ok = run(&mut env, "feasible(est, budget)");
    assert_eq!(ok.data()[0], 0.0, "vram over budget should be 0.0");
}

#[test]
fn feasible_zero_budget_skips_that_dimension() {
    let mut env = Environment::new();
    // Huge wall seconds, but wall budget is 0 (= skip).
    env.set(
        "est".into(),
        arr(vec![5], vec![1000.0, 4096.0, 128.0, 1e20, 1e10]),
    );
    env.set("budget".into(), arr(vec![3], vec![1e9, 1e9, 0.0]));
    let ok = run(&mut env, "feasible(est, budget)");
    assert_eq!(ok.data()[0], 1.0, "zero wall-budget should skip the check");
}

#[test]
fn feasible_wrong_shapes_error() {
    let mut env = Environment::new();
    env.set("est".into(), arr(vec![3], vec![1.0, 2.0, 3.0]));
    env.set("budget".into(), arr(vec![3], vec![0.0, 0.0, 0.0]));
    let stmts = parse(&lex("feasible(est, budget)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("est must be [5]");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("feasible"), "got: {msg}");
}

#[test]
fn feasible_composes_with_estimate_train() {
    // The intended guard pattern: compute an
    // estimate from a real model, then feed it to
    // feasible(...). Asserts the [5] / [3] shapes
    // flow through correctly across both builtins.
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    run(&mut env, "est = estimate_train(m, 100, 32, 1)");
    run(&mut env, "budget = [1000000.0, 1000000.0, 60.0]");
    let ok = run(&mut env, "feasible(est, budget)");
    assert_eq!(ok.data()[0], 1.0, "tiny linear easily fits");
}
