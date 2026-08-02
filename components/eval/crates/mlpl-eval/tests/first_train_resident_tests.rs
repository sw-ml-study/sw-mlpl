//! The FIRST resident training run in a process must match CPU
//! (saga E5 step 001 regression): the resident-optimizer gate used
//! to check `device_ops()` before anything had registered the
//! backend, so step 1 silently ran the CPU optimizer and the
//! resident path then reset the moments to zero -- a materially
//! different trajectory. This file must contain EXACTLY ONE test
//! so the train below is guaranteed to be the process's first
//! device use.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(env: &mut Environment, src: &str) -> mlpl_array::DenseArray {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env).expect("eval")
}

#[test]
fn very_first_resident_train_matches_cpu() {
    let run = |mlx: bool| -> Vec<f64> {
        let mut env = Environment::new();
        eval(
            &mut env,
            "m = chain(embed(16, 4, 0), linear(4, 16, 4)); \
             ids = [1, 2, 3, 1, 2]; Y = [2, 3, 1, 2, 3]; 0",
        );
        let body = "train 3 { momentum_sgd(cross_entropy(apply(m, ids), Y), m, 0.05, 0.9); \
                    cross_entropy(apply(m, ids), Y) }";
        let src = if mlx {
            format!("device(\"mlx\") {{ {body} }}; last_losses")
        } else {
            format!("{body}; last_losses")
        };
        eval(&mut env, &src).data().to_vec()
    };
    // MLX first: it must be the process's first device use.
    let mlx = run(true);
    let cpu = run(false);
    for (i, (c, m)) in cpu.iter().zip(&mlx).enumerate() {
        assert!(
            (c - m).abs() < 1e-4,
            "first-train parity, step {i}: cpu={c} mlx={m}"
        );
    }
}
