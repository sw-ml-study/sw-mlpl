//! Tests for the `device("target") { body }` scoped form
//! (Saga 14 step 004).
//!
//! Surface contract:
//!
//! - Parser emits `Expr::Device { target, body, span }` for any
//!   `device("...") { ... }` source. Unknown target strings parse
//!   fine -- only the evaluator decides what to do with them.
//! - `device("cpu") { body }` is always a value-yielding scoped
//!   form whose result is the value of the body's last statement.
//!   Same shapes, same labels, same numbers as running the body
//!   without the wrapper. Works on every host.
//! - `device("mlx") { body }` is the same on hosts where the
//!   `mlx` feature is unavailable: the evaluator emits a one-time
//!   warning and runs the body on CPU. On Apple Silicon with the
//!   `mlx` feature compiled in, eval routes ops through `mlpl-mlx-rt`
//!   but still produces the same shapes and labels (numeric values
//!   match within fp32 tolerance, asserted by the mlpl-mlx-rt parity
//!   tests, not here).
//! - Nesting works in either direction: `experiment { device { ...
//!   } }` and `device { experiment { ... } }` both compose
//!   correctly. An inner `device(...)` overrides the outer.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{Expr, lex, parse};

fn run(src: &str, env: &mut Environment) -> mlpl_array::DenseArray {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect("eval")
}

// -- Parser --

#[test]
fn parser_emits_device_variant_for_mlx() {
    let stmts = parse(&lex("device(\"mlx\") { x = 1 }").unwrap()).unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        Expr::Device { target, body, .. } => {
            assert_eq!(target, "mlx");
            assert_eq!(body.len(), 1);
        }
        other => panic!("expected Expr::Device, got {other:?}"),
    }
}

#[test]
fn parser_emits_device_variant_for_cpu() {
    let stmts = parse(&lex("device(\"cpu\") { x = 1 }").unwrap()).unwrap();
    matches!(
        &stmts[0],
        Expr::Device { target, .. } if target == "cpu"
    )
    .then_some(())
    .expect("expected Expr::Device with target=cpu");
}

#[test]
fn parser_accepts_unknown_target_strings() {
    // Any string parses; the evaluator decides whether to warn.
    parse(&lex("device(\"future-cuda\") { x = 1 }").unwrap()).unwrap();
}

#[test]
fn parser_rejects_missing_paren_around_target() {
    let r = parse(&lex("device \"mlx\" { x = 1 }").unwrap());
    assert!(r.is_err(), "device without parens should not parse");
}

#[test]
fn parser_rejects_non_string_target() {
    let r = parse(&lex("device(mlx) { x = 1 }").unwrap());
    assert!(r.is_err(), "device target must be a string literal");
}

#[test]
fn parser_rejects_missing_body() {
    let r = parse(&lex("device(\"mlx\")").unwrap());
    assert!(r.is_err(), "device requires a brace body");
}

// -- Evaluator: device("cpu") is always a no-op --

#[test]
fn cpu_block_yields_body_last_value() {
    let mut env = Environment::new();
    let r = run("device(\"cpu\") { iota(5) }", &mut env);
    assert_eq!(
        r.shape().dims(),
        &[5],
        "device('cpu') body should produce iota(5)'s shape"
    );
    assert_eq!(r.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn cpu_block_preserves_labels_through_boundary() {
    // A labeled tensor allocated inside device("cpu") is still
    // labeled when the block returns -- labels live in
    // `mlpl-core`, not in any device-specific runtime.
    let mut env = Environment::new();
    let r = run(
        "device(\"cpu\") { y : [batch, feat] = reshape(iota(6), [2, 3]) }",
        &mut env,
    );
    assert_eq!(r.shape().dims(), &[2, 3]);
    assert_eq!(
        r.labels(),
        Some(&[Some("batch".into()), Some("feat".into())][..])
    );
}

#[test]
fn cpu_block_assignments_visible_outside() {
    // Scoped form does not introduce a new variable scope: bindings
    // set inside survive the block. Mirrors `experiment { }`.
    let mut env = Environment::new();
    run("device(\"cpu\") { x = iota(3) }\n y = x", &mut env);
    let y = env
        .get("y")
        .expect("y was bound from x set inside the block");
    assert_eq!(y.data(), &[0.0, 1.0, 2.0]);
}

#[test]
fn empty_device_block_returns_scalar_zero() {
    let mut env = Environment::new();
    let r = run("device(\"cpu\") { }", &mut env);
    // Mirrors experiment{} on an empty body: yields a scalar 0
    // placeholder rather than erroring.
    assert_eq!(r.rank(), 0);
    assert_eq!(r.data(), &[0.0]);
}

// -- Evaluator: shapes + labels survive a device("mlx") boundary --

#[test]
fn mlx_block_preserves_shape_and_labels() {
    // Whether mlx feature is on or off, the shape and labels of
    // the body's last value must match what the CPU path would
    // produce. The mlpl-mlx-rt parity tests cover the numeric
    // tolerance separately; here we only assert the metadata
    // contract.
    let mut env_cpu = Environment::new();
    let cpu = run(
        "x : [batch, feat] = reshape(iota(6), [2, 3])\n device(\"cpu\") { x }",
        &mut env_cpu,
    );
    let mut env_mlx = Environment::new();
    let mlx = run(
        "x : [batch, feat] = reshape(iota(6), [2, 3])\n device(\"mlx\") { x }",
        &mut env_mlx,
    );
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
}

// -- Evaluator: nesting --

#[test]
fn experiment_around_device_records_one_run() {
    let mut env = Environment::new();
    run(
        "experiment \"with-device\" { device(\"cpu\") { loss_metric = 0.5 } }",
        &mut env,
    );
    let log = env.experiment_log();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].metrics.get("loss_metric").copied(), Some(0.5));
}

#[test]
fn device_around_experiment_records_one_run() {
    let mut env = Environment::new();
    run(
        "device(\"cpu\") { experiment \"inner\" { loss_metric = 0.25 } }",
        &mut env,
    );
    let log = env.experiment_log();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].metrics.get("loss_metric").copied(), Some(0.25));
}

// Saga 14 step 004: when the mlx feature is compiled in (and we
// are on Apple Silicon), ops inside `device("mlx") { }` actually
// route through `mlpl-mlx-rt`. We can observe this only indirectly
// at the eval layer -- the numeric output must agree with the
// CPU path within the documented fp32 tolerance. Same gate as
// `mlpl-mlx-rt`'s own parity tests so non-Apple CI stays green.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
#[test]
fn mlx_block_matmul_matches_cpu_within_fp32_tolerance() {
    const FP32_TOL: f64 = 1e-5;
    let src_cpu = "a = reshape(iota(8), [2, 4])\n\
                   b = reshape(iota(8), [4, 2])\n\
                   matmul(a, b)";
    let src_mlx = "a = reshape(iota(8), [2, 4])\n\
                   b = reshape(iota(8), [4, 2])\n\
                   device(\"mlx\") { matmul(a, b) }";
    let mut env_cpu = Environment::new();
    let mut env_mlx = Environment::new();
    let cpu = run(src_cpu, &mut env_cpu);
    let mlx = run(src_mlx, &mut env_mlx);
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    for (i, (m, c)) in mlx.data().iter().zip(cpu.data().iter()).enumerate() {
        assert!(
            (m - c).abs() <= FP32_TOL,
            "row-major elem {i}: mlx={m} cpu={c} diff={}",
            (m - c).abs()
        );
    }
}

#[test]
fn nested_device_inner_target_overrides_outer() {
    // The outer block sets device=mlx; the inner block sets
    // device=cpu. While inside the inner block, eval treats the
    // active device as cpu. After the inner block exits, the
    // active device pops back to mlx. We can only observe the
    // final body value here -- the dispatch fingerprint is left
    // to the parity tests in mlpl-mlx-rt -- but the result shape
    // must agree with the CPU path.
    let mut env = Environment::new();
    let r = run("device(\"mlx\") { device(\"cpu\") { iota(4) } }", &mut env);
    assert_eq!(r.shape().dims(), &[4]);
    assert_eq!(r.data(), &[0.0, 1.0, 2.0, 3.0]);
}
