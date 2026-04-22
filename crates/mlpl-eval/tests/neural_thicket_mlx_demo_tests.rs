//! Saga 20 step 005: validate the MLX variant loop of
//! `demos/neural_thicket_mlx.mlpl`.
//!
//! Runs a cut-down Neural Thickets sweep through both the CPU
//! path and the MLX path (variant loop wrapped in
//! `device("mlx") { ... }`) and asserts that:
//!
//! - Losses vector is shaped `[16]` on both and agrees
//!   elementwise within fp32 tolerance (matching the Saga 14
//!   tiny_lm_mlx_demo precedent).
//! - Heatmap reshape is `[4, 4]`.
//! - `argtop_k` on both paths returns 4 distinct in-range
//!   indices. Exact index equality across CPU and MLX is NOT
//!   asserted -- losses close within fp32 tolerance can
//!   reorder if two entries race at the tolerance boundary.
//! - Ensemble logits share the same shape as a single
//!   variant's forward output.
//!
//! Triple-gated on Apple Silicon + the `mlx` feature.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-3;

fn run(env: &mut Environment, src: &str) {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

fn assert_close(cpu: &DenseArray, mlx: &DenseArray, tol: f64, label: &str) {
    assert_eq!(cpu.shape(), mlx.shape(), "{label}: shape mismatch");
    for (i, (c, m)) in cpu.data().iter().zip(mlx.data().iter()).enumerate() {
        assert!(
            (c - m).abs() <= tol,
            "{label} elem {i}: cpu={c} mlx={m} diff={} tol={tol}",
            (c - m).abs()
        );
    }
}

/// Seed-aligned setup: same base model + same val tokens + same
/// sigma + same per-family seeds on both paths. Training is
/// skipped to keep the parity comparison as tight as possible
/// (the Saga 14 tiny_lm_mlx precedent allows training-under-MLX,
/// but Saga 20's thesis is the variant loop, not the training
/// step). Variant loop is in `device("mlx") { }` on the MLX path.
const BASE_PROGRAM: &str = "\
V = 32 ; d = 8 ; h = 1\n\
base = chain(embed(V, d, 0),\
             residual(chain(rms_norm(d), causal_attention(d, h, 1))),\
             residual(chain(rms_norm(d),\
                            linear(d, 16, 2),\
                            relu_layer(),\
                            linear(16, d, 3))),\
             rms_norm(d),\
             linear(d, V, 4))\n\
val_X = [1, 3, 5, 7, 2, 4, 6, 0, 9, 11, 13, 15, 2, 4, 6, 0]\n\
val_Y = [3, 5, 7, 2, 4, 6, 0, 1, 11, 13, 15, 2, 4, 6, 0, 1]\n\
sigma = 0.1\n\
losses = zeros([16])\n\
";

fn run_variant_loop(mlx: bool) -> (DenseArray, DenseArray, DenseArray) {
    let mut env = Environment::new();
    run(&mut env, BASE_PROGRAM);
    let sweep = "\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"all_layers\", sigma, i + 100);\
          losses = scatter(losses, i, cross_entropy(apply(v, val_X), val_Y))\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"attention_only\", sigma, i + 200);\
          losses = scatter(losses, 4 + i, cross_entropy(apply(v, val_X), val_Y))\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"mlp_only\", sigma, i + 300);\
          losses = scatter(losses, 8 + i, cross_entropy(apply(v, val_X), val_Y))\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"embed_and_head\", sigma, i + 400);\
          losses = scatter(losses, 12 + i, cross_entropy(apply(v, val_X), val_Y))\n\
        }\n\
        ens_logits = zeros(shape(apply(base, val_X)))\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"all_layers\", sigma, i + 100);\
          ens_logits = ens_logits + apply(v, val_X)\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"attention_only\", sigma, i + 200);\
          ens_logits = ens_logits + apply(v, val_X)\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"mlp_only\", sigma, i + 300);\
          ens_logits = ens_logits + apply(v, val_X)\n\
        }\n\
        for i in [0, 1, 2, 3] {\n\
          v = clone_model(base);\
          perturb_params(v, \"embed_and_head\", sigma, i + 400);\
          ens_logits = ens_logits + apply(v, val_X)\n\
        }\n\
        ens_logits = ens_logits * (1.0 / 16.0)\n\
    ";
    if mlx {
        let wrapped = format!(
            "device(\"mlx\") {{\n\
               to_device(base, \"mlx\");\
               to_device(val_X, \"mlx\");\
               {sweep}\n\
             }}"
        );
        run(&mut env, &wrapped);
    } else {
        run(&mut env, sweep);
    }
    let losses = env.get("losses").expect("losses bound").clone();
    let ens = env.get("ens_logits").expect("ens_logits bound").clone();
    // Shape of a single-variant forward as a reference.
    run(&mut env, "one_logits = apply(base, val_X)");
    let one = env.get("one_logits").expect("one_logits bound").clone();
    (losses, ens, one)
}

#[test]
fn neural_thicket_mlx_variant_loop_matches_cpu_within_fp32_tolerance() {
    let (cpu_losses, cpu_ens, cpu_one) = run_variant_loop(false);
    let (mlx_losses, mlx_ens, mlx_one) = run_variant_loop(true);

    // Core claim: identical seeds + identical base params + same
    // perturb deltas means per-variant losses agree within fp32
    // tolerance (CPU vs MLX differs only by per-kernel rounding).
    assert_eq!(cpu_losses.shape().dims(), &[16]);
    assert_eq!(mlx_losses.shape().dims(), &[16]);
    for v in cpu_losses.data().iter().chain(mlx_losses.data().iter()) {
        assert!(v.is_finite(), "losses must all be finite, got {v}");
    }
    assert_close(&cpu_losses, &mlx_losses, FP32_TOL, "variant losses");

    // Ensemble logits share the single-variant forward shape on
    // both paths and agree within tolerance.
    assert_eq!(cpu_ens.shape().dims(), cpu_one.shape().dims());
    assert_eq!(mlx_ens.shape().dims(), mlx_one.shape().dims());
    assert_close(&cpu_ens, &mlx_ens, FP32_TOL, "ensemble logits");

    // Heatmap shape is [4, 4] on both paths (trivially, since
    // losses are shape [16] and reshape is deterministic).
    let mut reshape_env = Environment::new();
    reshape_env.set("losses".into(), mlx_losses.clone());
    run(&mut reshape_env, "heat = reshape(losses, [4, 4])");
    assert_eq!(
        reshape_env.get("heat").unwrap().shape().dims(),
        &[4, 4],
        "heatmap shape must be [4, 4]"
    );

    // argtop_k returns 4 distinct in-range indices on both
    // paths. We do NOT assert exact equality of the index sets
    // because losses close within fp32 tolerance can reorder at
    // the boundary; the structural invariants are what matter.
    for (label, losses) in [("cpu", &cpu_losses), ("mlx", &mlx_losses)] {
        let mut env = Environment::new();
        env.set("losses".into(), losses.clone());
        run(&mut env, "best_idx = argtop_k(-1.0 * losses, 4)");
        let best_idx = env.get("best_idx").unwrap();
        assert_eq!(best_idx.shape().dims(), &[4]);
        let mut seen = std::collections::HashSet::new();
        for &idx_f in best_idx.data() {
            let idx = idx_f as usize;
            assert!(idx < 16, "{label}: best_idx entry {idx} out of range");
            assert!(
                seen.insert(idx),
                "{label}: best_idx should have distinct entries"
            );
        }
    }
}

#[test]
fn neural_thicket_mlx_demo_file_parses() {
    // Belt-and-braces: the shipped demo file itself parses
    // cleanly (variant-loop parity is exercised above; this
    // validates the exact bytes on disk).
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/neural_thicket_mlx.mlpl"
    ))
    .expect("read demos/neural_thicket_mlx.mlpl");
    let tokens = lex(&src).expect("demo lexes");
    let stmts = parse(&tokens).expect("demo parses");
    assert!(!stmts.is_empty(), "demo should have at least one statement");
}
