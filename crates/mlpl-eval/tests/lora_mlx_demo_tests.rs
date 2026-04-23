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
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-3;

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

    // Every student param (adapters + frozen base) agrees
    // elementwise.
    assert_eq!(
        cpu_params.keys().len(),
        mlx_params.keys().len(),
        "student should own the same number of params on both paths"
    );
    for (name, cpu_vals) in &cpu_params {
        let mlx_vals = mlx_params
            .get(name)
            .unwrap_or_else(|| panic!("mlx path missing param '{name}'"));
        assert_close_slice(cpu_vals, mlx_vals, FP32_TOL, &format!("param '{name}'"));
    }
}

#[test]
fn lora_mlx_finetune_leaves_frozen_base_bit_identical_on_both_paths() {
    // The frozen-params rule is a property of the optimizer
    // dispatch, which lives in grad.rs (CPU-side). Both paths
    // must honor it identically -- freezing is backend-
    // independent by design.
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
        "/../../demos/lora_finetune_mlx.mlpl"
    ))
    .expect("read demos/lora_finetune_mlx.mlpl");
    let tokens = lex(&src).expect("demo lexes");
    let stmts = parse(&tokens).expect("demo parses");
    assert!(!stmts.is_empty(), "demo should have at least one statement");
}
