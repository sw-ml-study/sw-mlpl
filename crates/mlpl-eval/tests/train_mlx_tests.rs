//! Saga 14 step 007: optimizers + `train { }` parity inside
//! `device("mlx") { }`.
//!
//! Step 006 wired `grad(expr, wrt)` to re-materialize tape forward
//! values through `mlpl-mlx` before CPU backward formulas compute
//! the gradient. Adam and momentum_sgd both call `eval_grad`
//! internally, so their parameter updates ride on those
//! MLX-rounded gradients. The step-007 invariant: an Adam or
//! momentum_sgd step inside `device("mlx") { }` yields parameter
//! values that match the all-CPU path within fp32 tolerance, and
//! `train N { adam(...) }` yields the same `last_losses` vector
//! on both devices.
//!
//! Design note: `OptimizerState` buffers (Adam `m`/`v`,
//! momentum_sgd `v`) stay CPU-resident `DenseArray` values --
//! MLX-resident optimizer state is a perf optimization, not a
//! correctness requirement, and the values stored in those
//! buffers are computed from MLX-rounded gradients regardless.
//! Documented here and in the commit so step 008's Tiny LM
//! benchmark knows it is NOT yet doing all the Adam arithmetic
//! on MLX; a follow-up step can lift the buffers without
//! changing numeric behaviour.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-4;

fn run(src: &str) -> (DenseArray, Environment) {
    let mut env = Environment::new();
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let r = eval_program(&stmts, &mut env).expect("eval");
    (r, env)
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

// ---- one Adam step parity ----

#[test]
fn one_adam_step_on_linear_matches_cpu_within_fp32_tolerance() {
    // Same fixed seeds, same hyperparams. After one Adam step the
    // linear's W param should be updated by the same delta on both
    // devices within fp32 tolerance.
    let setup = "m = linear(3, 2, 0)\n \
                 X = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0], [2.0, 0.0, 1.0]]\n \
                 Y = [0, 1, 0]";
    let step = "adam(cross_entropy(apply(m, X), Y), m, 0.01, 0.9, 0.999, 0.00000001)";

    let cpu_src = format!("{setup}\n {step}\n __linear_W_0");
    let mlx_src = format!(
        "device(\"mlx\") {{ m = linear(3, 2, 0) }}\n \
         X = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0], [2.0, 0.0, 1.0]]\n \
         to_device(X, \"mlx\")\n \
         Y = [0, 1, 0]\n \
         device(\"mlx\") {{ {step} }}\n \
         __linear_W_0"
    );
    let (cpu_w, _) = run(&cpu_src);
    let (mlx_w, _) = run(&mlx_src);
    assert_close(&cpu_w, &mlx_w, FP32_TOL, "adam W update");
}

#[test]
fn one_momentum_sgd_step_on_linear_matches_cpu_within_fp32_tolerance() {
    let setup = "m = linear(3, 2, 0)\n \
                 X = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0], [2.0, 0.0, 1.0]]\n \
                 Y = [0, 1, 0]";
    let step = "momentum_sgd(cross_entropy(apply(m, X), Y), m, 0.01, 0.9)";

    let cpu_src = format!("{setup}\n {step}\n __linear_W_0");
    let mlx_src = format!(
        "device(\"mlx\") {{ m = linear(3, 2, 0) }}\n \
         X = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0], [2.0, 0.0, 1.0]]\n \
         to_device(X, \"mlx\")\n \
         Y = [0, 1, 0]\n \
         device(\"mlx\") {{ {step} }}\n \
         __linear_W_0"
    );
    let (cpu_w, _) = run(&cpu_src);
    let (mlx_w, _) = run(&mlx_src);
    assert_close(&cpu_w, &mlx_w, FP32_TOL, "momentum_sgd W update");
}

// ---- train { } parity: same last_losses on both devices ----

#[test]
fn train_3_steps_last_losses_match_cpu_within_fp32_tolerance() {
    // A Saga 13 Tiny LM-shaped micro-slice: embed -> linear -> CE.
    // Three training steps; last_losses must agree on both
    // devices within fp32 tolerance.
    let model = "chain(embed(6, 4, 0), rms_norm(4), linear(4, 6, 2))";
    let data_cpu = "X = [1, 3, 5, 2]\n Y = [2, 4, 0, 1]";
    let data_mlx = "X = [1, 3, 5, 2]\n to_device(X, \"mlx\")\n Y = [2, 4, 0, 1]";

    let cpu_src = format!(
        "m = {model}\n {data_cpu}\n \
         train 3 {{ \
           adam(cross_entropy(apply(m, X), Y), m, 0.01, 0.9, 0.999, 0.00000001); \
           cross_entropy(apply(m, X), Y) \
         }}\n last_losses"
    );
    let mlx_src = format!(
        "device(\"mlx\") {{ m = {model} }}\n {data_mlx}\n \
         device(\"mlx\") {{ \
           train 3 {{ \
             adam(cross_entropy(apply(m, X), Y), m, 0.01, 0.9, 0.999, 0.00000001); \
             cross_entropy(apply(m, X), Y) \
           }} \
         }}\n last_losses"
    );
    let (cpu_losses, _) = run(&cpu_src);
    let (mlx_losses, _) = run(&mlx_src);
    assert_close(
        &cpu_losses,
        &mlx_losses,
        FP32_TOL,
        "last_losses after 3 steps",
    );
}

// ---- schedulers (pure scalar math, no dispatch) ----

#[test]
fn schedules_work_inside_mlx_block() {
    // cosine_schedule and linear_warmup are scalar-in, scalar-out
    // and pass through dispatched_call to `mlpl_runtime::call_builtin`.
    // They do not need a device-specific path; this test just
    // witnesses that the MLX scope does not break them.
    let src = "device(\"mlx\") { \
               a = cosine_schedule(50, 100, 0.0001, 0.01); \
               b = linear_warmup(50, 100, 0.01); \
               [a, b] \
               }";
    let (r, _) = run(src);
    assert_eq!(r.shape().dims(), &[2]);
    // Midway through cosine schedule (50/100) gives lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi/2)) = midpoint.
    // Here that's 0.0001 + 0.5*0.0099*(1+0) = 0.00505.
    assert!((r.data()[0] - 0.00505).abs() < 1e-9);
    // Linear warmup halfway through gives 0.5 * lr = 0.005.
    assert!((r.data()[1] - 0.005).abs() < 1e-9);
}

// ---- experiment + _metric capture inside device("mlx") ----

#[test]
fn experiment_metric_capture_works_inside_mlx_block() {
    // Same program run twice: once with device("mlx") wrapping
    // experiment, once without. Both should record one run with
    // the same metric value (scalar CPU write, no dispatch
    // needed -- the test is only that the metric mechanism is
    // not broken by the MLX scope).
    let src_cpu = "experiment \"inside-cpu\" { loss_metric = 0.5 }";
    let src_mlx = "device(\"mlx\") { experiment \"inside-mlx\" { loss_metric = 0.5 } }";
    let (_, env_cpu) = run(src_cpu);
    let (_, env_mlx) = run(src_mlx);
    let cpu_rec = env_cpu.experiment_log().last().unwrap();
    let mlx_rec = env_mlx.experiment_log().last().unwrap();
    assert_eq!(
        cpu_rec.metrics.get("loss_metric").copied(),
        mlx_rec.metrics.get("loss_metric").copied()
    );
    assert_eq!(mlx_rec.metrics.get("loss_metric").copied(), Some(0.5));
}
