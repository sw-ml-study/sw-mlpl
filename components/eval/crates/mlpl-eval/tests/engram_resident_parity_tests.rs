//! Engram on the resident MLX tape (saga E5 step 001): the frozen
//! hash contract stays bit-exact under a device block, addressing
//! statistics agree between CPU and resident training, and the
//! engram-in-chain loss trajectory tracks the CPU tape at fp32
//! tolerance over a real multi-step run.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> mlpl_array::DenseArray {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env).expect("eval")
}

/// Full engram-in-chain fixture: byte-ish vocab 16, d=4, one
/// attention block, engram(4, [2], 1, 8, 4, 7) after attention.
const SETUP: &str = "emb = embed(16, 4, 0); \
     att = residual(chain(rms_norm(4), causal_attention(4, 2, 1))); \
     head = chain(rms_norm(4), linear(4, 16, 4)); \
     e = engram(4, [2], 1, 8, 4, 7); \
     m = chain(emb, att, e, head); \
     ids = [1, 2, 3, 1, 2]; Y = [2, 3, 1, 2, 3]";
const STEP: &str = "adam(cross_entropy(apply(m, ids), Y), m, 0.02, 0.9, 0.999, 0.00000001)";

#[test]
fn hash_contract_is_bit_exact_under_a_device_block() {
    let mut cpu = Environment::new();
    let a = eval(&mut cpu, "ngram_hash([10, 20, 30, 40], [2, 3], 4, 1024, 7)");
    let mut mlx = Environment::new();
    let b = eval(
        &mut mlx,
        "device(\"mlx\") { ngram_hash([10, 20, 30, 40], [2, 3], 4, 1024, 7) }",
    );
    assert_eq!(a.shape().dims(), b.shape().dims());
    // Bit-exact: the addressing contract is frozen across backends.
    assert_eq!(a.data(), b.data(), "hash contract must not drift");
}

/// Train the same engram chain N steps on CPU and resident-MLX;
/// return (last_losses, nonzero_rows, max_row_norm).
fn train(mlx: bool, steps: usize) -> (Vec<f64>, f64, f64) {
    let mut env = Environment::new();
    eval(&mut env, SETUP);
    let body = format!("train {steps} {{ {STEP}; cross_entropy(apply(m, ids), Y) }}");
    let src = if mlx {
        format!("device(\"mlx\") {{ {body} }}; last_losses")
    } else {
        format!("{body}; last_losses")
    };
    let losses = eval(&mut env, &src).data().to_vec();
    eval(&mut env, "s = engram_stats(e, ids); 0");
    let nz = eval(&mut env, "s.nonzero_rows").data()[0];
    let rn = eval(&mut env, "s.max_row_norm").data()[0];
    (losses, nz, rn)
}

#[test]
fn ten_step_engram_chain_trajectory_and_stats_match_cpu() {
    let (cpu_l, cpu_nz, cpu_rn) = train(false, 10);
    let (mlx_l, mlx_nz, mlx_rn) = train(true, 10);
    assert_eq!(cpu_l.len(), 10);
    let mut max_drift = 0.0f64;
    for (c, m) in cpu_l.iter().zip(&mlx_l) {
        max_drift = max_drift.max((c - m).abs());
    }
    println!("10-step engram-chain max loss drift: {max_drift:.6}");
    assert!(max_drift < 1e-3, "trajectory drift {max_drift} over bound");
    // Addressing is index-determined, so the set of written rows is
    // EXACTLY equal; the gate value is fp32-tolerant.
    assert_eq!(cpu_nz, mlx_nz, "addressed-row count must match exactly");
    assert!(
        (cpu_rn - mlx_rn).abs() < 1e-3,
        "max_row_norm parity: {cpu_rn} vs {mlx_rn}"
    );
    // And it actually learns.
    assert!(mlx_l[9] < mlx_l[0], "resident engram training converges");
}

#[test]
fn duplicate_address_scatter_add_matches_cpu_gradients() {
    // ids [1, 2, 3, 1, 2] repeat the (1, 2) bigram, so two rows of
    // the selection matrix address the same memory row. The
    // resident backward (sel^T @ upstream) must ACCUMULATE those
    // contributions exactly like the CPU scatter-ADD kernel.
    let run = |mlx: bool| -> Vec<f64> {
        let mut env = Environment::new();
        eval(&mut env, SETUP);
        let mem = {
            let name = env.get_model("e").unwrap().params()[0].clone();
            // Non-zero memory so the value/gate path is grad-sensitive.
            let cur = env.get(&name).unwrap().clone();
            let filled = mlpl_array::DenseArray::new(
                cur.shape().clone(),
                (0..cur.data().len())
                    .map(|i| 0.03 + i as f64 * 0.01)
                    .collect(),
            )
            .unwrap();
            env.set(name.clone(), filled);
            name
        };
        let g = format!("grad(cross_entropy(apply(m, ids), Y), {mem})");
        let src = if mlx {
            format!("device(\"mlx\") {{ {g} }}")
        } else {
            g
        };
        eval(&mut env, &src).data().to_vec()
    };
    let (cpu, mlx) = (run(false), run(true));
    assert_eq!(cpu.len(), mlx.len());
    let nonzero = cpu.iter().filter(|v| v.abs() > 1e-12).count();
    assert!(nonzero > 0, "memory gradient must be non-trivial");
    for (i, (c, m)) in cpu.iter().zip(&mlx).enumerate() {
        assert!(
            (c - m).abs() < 1e-5,
            "scatter-add parity at {i}: cpu={c} mlx={m}"
        );
    }
}
