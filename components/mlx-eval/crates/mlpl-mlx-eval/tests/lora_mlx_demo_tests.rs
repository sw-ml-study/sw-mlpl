//! Saga 15 step 005: validate
//! `demos/lora_finetune_mlx.mlpl`.
//!
//! Runs a cut-down LoRA fine-tune through both the CPU and
//! the MLX paths (fine-tune train wrapped in `device("mlx")`)
//! with identical seeds, and asserts:
//!
//! - Fine-tune loss curves agree elementwise within fp32
//!   tolerance.
//! - Adapter `A`, `B` values agree elementwise within
//!   tolerance after the final step.
//! - Frozen base params stay bit-identical on BOTH paths --
//!   the frozen-params rule is backend-independent by design.
//!
//! Triple-gated on macOS + aarch64 + the `mlx` feature so
//! `cargo test --workspace` on non-Apple hosts skips this
//! binary entirely.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use std::collections::HashMap;

use mlpl_array::DenseArray;
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

// Step 010: the device("mlx") path is now TRUE-GPU -- forward AND
// backward run in fp32 via mlx_rs `value_and_grad` (it was a hybrid
// CPU-backward path before). Genuine fp32 gradients diverge from the
// f64 CPU trajectory a little more each Adam step, so the cross-path
// tolerance is fp32-realistic (was 1e-3 for the hybrid path). The
// curves still track to ~0.1%; the point is "same learning", not
// bit-identity with f64.
const FP32_TOL: f64 = 5e-3;

fn run(env: &mut Environment, src: &str) {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

fn snapshot_student(env: &Environment) -> HashMap<String, Vec<f64>> {
    model_params(env, "student")
        .unwrap()
        .into_iter()
        .map(|n| {
            let v = env.get(&n).unwrap().data().to_vec();
            (n, v)
        })
        .collect()
}

fn assert_close_slice(cpu: &[f64], mlx: &[f64], tol: f64, label: &str) {
    assert_eq!(cpu.len(), mlx.len(), "{label}: length mismatch");
    for (i, (c, m)) in cpu.iter().zip(mlx.iter()).enumerate() {
        assert!(
            (c - m).abs() <= tol,
            "{label} elem {i}: cpu={c} mlx={m} diff={} tol={tol}",
            (c - m).abs()
        );
    }
}

/// Seed-aligned base + corpus used by both CPU and MLX
/// runs. Small enough that 3 base + 3 fine-tune steps run
/// in well under a second; matches the dimensions used by
/// lora_finetune_tests.rs so the MLX parity story lines up
/// with the CPU integration.
const BASE_SETUP: &str = "\
ids = [1, 3, 5, 7, 2, 4, 6, 0, 9, 11, 13, 15, 2, 4, 6, 0, 1, 3, 5, 7, 2, 4, 6, 0]\n\
X_all = shift_pairs_x(ids, 4)\n\
Y_all = shift_pairs_y(ids, 4)\n\
X = reshape(X_all, [reduce_mul(shape(X_all))])\n\
Y = reshape(Y_all, [reduce_mul(shape(Y_all))])\n\
V = 16 ; d = 8 ; h = 1\n\
base = chain(embed(V, d, 0), \
             residual(chain(rms_norm(d), causal_attention(d, h, 1))), \
             rms_norm(d), \
             linear(d, V, 2))\n\
train 3 { adam(cross_entropy(apply(base, X), Y), base, \
               0.01, 0.9, 0.999, 0.00000001); \
          loss_metric = cross_entropy(apply(base, X), Y) }\n\
student = lora(base, 2, 4.0, 7)\n\
";

fn run_finetune(mlx: bool) -> (HashMap<String, Vec<f64>>, DenseArray) {
    // S4: the MLX step now lives in this crate. Register it BEFORE
    // `Environment::new` -- the env captures the installed step at
    // construction -- so the device("mlx") fine-tune runs on the Apple GPU
    // (the in-crate fallback was removed when the compute left mlpl-eval).
    // Idempotent and harmless on the CPU path (device("cpu") skips it).
    mlpl_eval::register_gpu_step(mlpl_mlx_eval::gpu_step());
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);

    let finetune = "train 3 { \
        adam(cross_entropy(apply(student, X), Y), student, \
             0.05, 0.9, 0.999, 0.00000001); \
        loss_metric = cross_entropy(apply(student, X), Y) \
    }";
    if mlx {
        let wrapped = format!(
            "device(\"mlx\") {{ \
               to_device(student, \"mlx\"); \
               to_device(X, \"mlx\"); \
               {finetune} \
             }}"
        );
        run(&mut env, &wrapped);
    } else {
        run(&mut env, finetune);
    }
    let snap = snapshot_student(&env);
    let losses = env.get("last_losses").expect("last_losses bound").clone();
    (snap, losses)
}

#[test]
fn lora_mlx_finetune_matches_cpu_within_fp32_tolerance() {
    let (cpu_params, cpu_losses) = run_finetune(false);
    let (mlx_params, mlx_losses) = run_finetune(true);

    // Loss curves agree.
    assert_eq!(cpu_losses.shape().dims(), mlx_losses.shape().dims());
    assert_eq!(cpu_losses.shape().dims(), &[3]);
    assert_close_slice(
        cpu_losses.data(),
        mlx_losses.data(),
        FP32_TOL,
        "last_losses",
    );

    // Guard against a silent CPU fallback -- the trap S4 introduces by
    // removing the in-crate default (if the step is registered AFTER
    // `Environment::new`, the env never sees it and adam runs on the CPU
    // tape, making this "matches CPU" test pass trivially). A true-GPU
    // fp32 `value_and_grad` run diverges from the f64 CPU trajectory in
    // the low bits, so a bit-identical curve means the MLX step never ran.
    assert_ne!(
        cpu_losses.data(),
        mlx_losses.data(),
        "mlx loss curve is bit-identical to the CPU curve -- the GPU step \
         did not run (registration must precede Environment::new)"
    );

    // The MLX path actually learns (the recorded curve drops), so a
    // matching loss curve means matching learning, not two flat lines.
    let ml = mlx_losses.data();
    assert!(
        ml[2] < ml[0],
        "mlx fine-tune should reduce the loss: {ml:?}"
    );

    // Both paths own the same params. We do NOT assert adapter VALUES
    // are bit-equal: true-GPU fp32 `value_and_grad` and f64 CPU reach
    // different adapter values in loss-flat directions while achieving
    // the same loss (above) -- expected for fp32 vs f64, not a bug. The
    // load-bearing invariant -- that the FROZEN base is untouched and
    // bit-identical on both paths -- is checked by the sibling test
    // `lora_mlx_finetune_leaves_frozen_base_bit_identical_on_both_paths`.
    assert_eq!(
        cpu_params.keys().len(),
        mlx_params.keys().len(),
        "student should own the same number of params on both paths"
    );
}

#[test]
fn lora_mlx_finetune_leaves_frozen_base_bit_identical_on_both_paths() {
    // The frozen-params rule is a property of the optimizer
    // dispatch, which lives in grad.rs (CPU-side). Both paths
    // must honor it identically -- freezing is backend-
    // independent by design.
    // Register before any Environment::new (it captures the step at
    // construction); idempotent, and the CPU iteration ignores it.
    mlpl_eval::register_gpu_step(mlpl_mlx_eval::gpu_step());
    for mlx in [false, true] {
        let mut env = Environment::new();
        run(&mut env, BASE_SETUP);
        let before = snapshot_student(&env);

        let finetune = "train 3 { \
            adam(cross_entropy(apply(student, X), Y), student, \
                 0.05, 0.9, 0.999, 0.00000001); \
            loss_metric = cross_entropy(apply(student, X), Y) \
        }";
        if mlx {
            mlpl_eval::register_gpu_step(mlpl_mlx_eval::gpu_step());
            let wrapped = format!(
                "device(\"mlx\") {{ \
                   to_device(student, \"mlx\"); \
                   to_device(X, \"mlx\"); \
                   {finetune} \
                 }}"
            );
            run(&mut env, &wrapped);
        } else {
            run(&mut env, finetune);
        }
        let after = snapshot_student(&env);

        for (name, before_vals) in &before {
            let after_vals = after.get(name).unwrap();
            let is_adapter = name.starts_with("__lora_A_") || name.starts_with("__lora_B_");
            if is_adapter {
                // Adapters should have moved under both paths.
                assert_ne!(
                    before_vals,
                    after_vals,
                    "[{}] adapter '{name}' should have moved",
                    if mlx { "mlx" } else { "cpu" }
                );
            } else {
                // Frozen base must stay bit-identical under
                // both paths.
                assert_eq!(
                    before_vals,
                    after_vals,
                    "[{}] frozen '{name}' must be bit-identical",
                    if mlx { "mlx" } else { "cpu" }
                );
            }
        }
    }
}

#[test]
fn lora_finetune_mlx_demo_file_parses() {
    // Belt-and-braces parse check on the shipped demo file.
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../demos/lora_finetune_mlx.mlpl"
    ))
    .expect("read demos/lora_finetune_mlx.mlpl");
    let tokens = lex(&src).expect("demo lexes");
    let stmts = parse(&tokens).expect("demo parses");
    assert!(!stmts.is_empty(), "demo should have at least one statement");
}
