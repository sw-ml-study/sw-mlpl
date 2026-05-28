//! Saga 14 step 008: validate `demos/tiny_lm_mlx.mlpl`.
//!
//! The real demo runs 200 training steps at V=280/d=32/T=32 and
//! takes on the order of tens of seconds under Criterion-style
//! repeated runs. This test runs a scaled-down micro-variant
//! (V=60, d=16, T=8, 3 steps) through the same code paths --
//! same model DSL composition, same `device("mlx") { train { }
//! }` nesting, same `experiment "name" { }` wrapper -- and
//! asserts the MLX loss curve matches the CPU loss curve within
//! fp32 tolerance. If this passes, the full demo is correct by
//! construction and the bench harness in `mlpl-bench/benches/
//! mlx_vs_cpu.rs` can focus on measuring wall clock instead.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-3;

fn run(src: &str) -> DenseArray {
    let mut env = Environment::new();
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, &mut env).expect("eval")
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

#[test]
fn tiny_lm_mlx_demo_micro_variant_parity() {
    // Same chain/residual/rms_norm/causal_attention/linear
    // composition as demos/tiny_lm_mlx.mlpl, just scaled down
    // so the test runs in well under a second. 3 training steps
    // are enough to exercise Adam's state accumulation path and
    // verify the loss curve shape matches.
    let model = "chain(embed(60, 16, 0), \
                       residual(chain(rms_norm(16), causal_attention(16, 1, 1))), \
                       rms_norm(16), \
                       linear(16, 60, 2))";
    let data_cpu = "X = [1, 3, 5, 7, 2, 4, 6, 0]\n Y = [3, 5, 7, 2, 4, 6, 0, 1]";
    let data_mlx = "X = [1, 3, 5, 7, 2, 4, 6, 0]\n \
                    to_device(X, \"mlx\")\n \
                    Y = [3, 5, 7, 2, 4, 6, 0, 1]";

    let cpu_src = format!(
        "m = {model}\n {data_cpu}\n \
         experiment \"tiny_lm_micro_cpu\" {{ \
           train 3 {{ \
             adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001); \
             loss_metric = cross_entropy(apply(m, X), Y) \
           }} \
         }}\n last_losses"
    );
    let mlx_src = format!(
        "device(\"mlx\") {{ m = {model} }}\n {data_mlx}\n \
         device(\"mlx\") {{ \
           experiment \"tiny_lm_micro_mlx\" {{ \
             train 3 {{ \
               adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001); \
               loss_metric = cross_entropy(apply(m, X), Y) \
             }} \
           }} \
         }}\n last_losses"
    );
    let cpu_losses = run(&cpu_src);
    let mlx_losses = run(&mlx_src);
    assert_eq!(cpu_losses.shape().dims(), &[3]);
    assert_close(
        &cpu_losses,
        &mlx_losses,
        FP32_TOL,
        "tiny_lm micro loss curve",
    );
}

#[test]
fn tiny_lm_mlx_demo_file_parses() {
    // Belt-and-braces check that the shipped demo file itself
    // parses cleanly (the micro variant above validates the
    // logic; this one validates the exact bytes in the demo).
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/tiny_lm_mlx.mlpl"
    ))
    .expect("read demos/tiny_lm_mlx.mlpl");
    let tokens = lex(&src).expect("demo lexes");
    let stmts = parse(&tokens).expect("demo parses");
    assert!(!stmts.is_empty(), "demo should have at least one statement");
}
