//! The resident optimizer path (saga E4 step 006): under
//! device("mlx"), adam/momentum run ONE tape per step and keep
//! weights + moments resident across the whole loop -- pinned here
//! by inspecting the optimizer state after a multi-step train.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-4;

fn eval(env: &mut Environment, src: &str) -> mlpl_array::DenseArray {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env).expect("eval")
}

const SETUP: &str = "m = linear(3, 2, 0); \
     X = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0], [2.0, 0.0, 1.0]]; \
     Y = [0, 1, 0]";
const STEP: &str = "adam(cross_entropy(apply(m, X), Y), m, 0.01, 0.9, 0.999, 0.00000001)";

#[test]
fn weights_and_moments_stay_resident_across_the_loop() {
    let mut env = Environment::new();
    eval(&mut env, SETUP);
    eval(
        &mut env,
        &format!("device(\"mlx\") {{ train 3 {{ {STEP} }} }}"),
    );
    let w_name = env.get_model("m").unwrap().params()[0].clone();
    let st = &env.optim_state;
    for slot in [
        ("adam".to_string(), w_name.clone(), "m".to_string()),
        ("adam".to_string(), w_name.clone(), "v".to_string()),
        ("resident".to_string(), w_name.clone(), "w".to_string()),
    ] {
        let h = st.resident.get(&slot).expect("resident slot present");
        assert!(h.is_dev(), "{slot:?} must live on the device");
        // Superseded host moments are dropped.
        assert!(
            !st.buffers.contains_key(&slot),
            "{slot:?} host buffer dropped"
        );
    }
    // The host mirror is fresh: witness matches the live var.
    let live_ptr = env.get(&w_name).unwrap().data().as_ptr() as usize;
    assert_eq!(st.resident_witness.get(&w_name), Some(&live_ptr));
}

#[test]
fn three_step_trajectory_matches_cpu_within_tolerance() {
    let run = |mlx: bool| -> Vec<f64> {
        let mut env = Environment::new();
        eval(&mut env, SETUP);
        let body = format!("train 3 {{ {STEP}; cross_entropy(apply(m, X), Y) }}");
        let src = if mlx {
            format!("device(\"mlx\") {{ {body} }}; last_losses")
        } else {
            format!("{body}; last_losses")
        };
        eval(&mut env, &src).data().to_vec()
    };
    let (cpu, mlx) = (run(false), run(true));
    assert_eq!(cpu.len(), mlx.len());
    for (i, (c, m)) in cpu.iter().zip(&mlx).enumerate() {
        assert!((c - m).abs() < FP32_TOL, "step {i}: cpu={c} mlx={m}");
    }
}

#[test]
fn frozen_base_engram_training_works_resident() {
    // The E3 frozen-base story under device("mlx"): base params stay
    // bit-identical, only the engram learns, loss drops -- now with
    // the engram's memory + moments resident.
    let run = |mlx: bool| -> (f64, f64) {
        let mut env = Environment::new();
        eval(
            &mut env,
            "emb = embed(16, 4, 0); att = residual(chain(rms_norm(4), causal_attention(4, 2, 1))); \
             head = chain(rms_norm(4), linear(4, 16, 4)); e = engram(4, [2], 1, 8, 4, 7); \
             m2 = chain(emb, att, e, head); ids = [1, 2, 3]; \
             freeze(emb); freeze(att); freeze(head); \
             y = apply(m2, ids) + 1; 0",
        );
        let step = "adam(mean((apply(m2, ids) - y) * (apply(m2, ids) - y)), m2, 0.05, 0.9, 0.999, 0.00000001)";
        let body = format!(
            "l0 = mean((apply(m2, ids) - y) * (apply(m2, ids) - y)); train 10 {{ {step} }}"
        );
        let src = if mlx {
            format!("device(\"mlx\") {{ {body} }}; 0")
        } else {
            format!("{body}; 0")
        };
        eval(&mut env, &src);
        let l0 = eval(&mut env, "l0").data()[0];
        let l1 = eval(
            &mut env,
            "mean((apply(m2, ids) - y) * (apply(m2, ids) - y))",
        )
        .data()[0];
        (l0, l1)
    };
    let (c0, c1) = run(false);
    let (m0, m1) = run(true);
    assert!(c1 < c0 * 0.9, "cpu frozen-base training reduces loss");
    assert!(m1 < m0 * 0.9, "resident frozen-base training reduces loss");
    assert!(
        (c1 - m1).abs() < 1e-2,
        "trajectories comparable: {c1} vs {m1}"
    );
}

const TINY_LM_SETUP: &str = "m = chain(embed(60, 16, 0), \
     residual(chain(rms_norm(16), causal_attention(16, 1, 1))), \
     rms_norm(16), linear(16, 60, 2)); \
     X = [1, 3, 5, 7, 2, 4, 6, 0]; Y = [3, 5, 7, 2, 4, 6, 0, 1]";

#[test]
fn thirty_step_tiny_lm_trajectory_stays_within_tolerance() {
    // The step-008 gate: the bench-shaped tiny LM (V=60, d=16, T=8,
    // 1-head causal attention) trained 30 steps resident must track
    // the CPU loss trajectory. fp32 drift compounds over 30 steps
    // (measured max drift ~1e-6 at this scale); 1e-3 gives wide
    // headroom while catching any semantic divergence (a wrong
    // gradient blows past it in 2-3 steps).
    let step = "adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001)";
    let run = |mlx: bool| -> Vec<f64> {
        let mut env = Environment::new();
        eval(&mut env, TINY_LM_SETUP);
        let body = format!("train 30 {{ {step}; cross_entropy(apply(m, X), Y) }}");
        let src = if mlx {
            format!("device(\"mlx\") {{ {body} }}; last_losses")
        } else {
            format!("{body}; last_losses")
        };
        eval(&mut env, &src).data().to_vec()
    };
    let (cpu, mlx) = (run(false), run(true));
    assert_eq!(cpu.len(), 30);
    assert_eq!(mlx.len(), 30);
    let mut max_drift = 0.0f64;
    for (c, m) in cpu.iter().zip(&mlx) {
        max_drift = max_drift.max((c - m).abs());
    }
    println!("30-step tiny-LM trajectory max drift: {max_drift:.6}");
    assert!(max_drift < 1e-3, "trajectory drift {max_drift} over bound");
    // And it actually trains: the loss must fall materially.
    assert!(mlx[29] < mlx[0] * 0.8, "resident training converges");
}
