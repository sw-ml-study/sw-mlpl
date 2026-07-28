//! Tests for Model DSL dispatch through the active device
//! (Saga 14 step 005).
//!
//! Contract:
//!
//! - `to_device(x, "mlx")` / `to_device(x, "cpu")` records the
//!   target on the environment's per-tensor map. Value is
//!   unchanged. Round-trip cpu -> mlx -> cpu returns bit-for-bit
//!   the original array; mlx -> cpu -> mlx is bounded by the
//!   `mlpl-mlx-rt` fp32 tolerance (the value path is CPU-resident
//!   until later phases add real device-backed storage).
//! - `apply(model, X)` raises `EvalError::DeviceMismatch` when
//!   the model's params and `X` disagree on device placement.
//! - Inside `device("mlx") { }`, `apply(model, X)` routes
//!   matmul/softmax/add through `mlpl-mlx-rt`. Output shape and
//!   labels must match the CPU path exactly; numeric values
//!   match within the documented fp32 tolerance.
//! - The Tiny LM forward assembled from the Saga 11 layers
//!   (embed + residual + rms_norm + causal_attention + linear)
//!   produces the same logits in `device("mlx") { }` as in
//!   `device("cpu") { }` within fp32 tolerance, proving the
//!   step's end-to-end goal.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> mlpl_array::DenseArray {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect("eval")
}

fn run_err(src: &str, env: &mut Environment) -> EvalError {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect_err("should fail")
}

// -- to_device round-trip --

#[test]
fn to_device_cpu_to_mlx_to_cpu_is_bitwise_identical() {
    let mut env = Environment::new();
    run(
        "x = reshape(iota(6), [2, 3])\n\
         to_device(x, \"mlx\")\n\
         to_device(x, \"cpu\")",
        &mut env,
    );
    let x = env.get("x").expect("x bound");
    assert_eq!(x.shape().dims(), &[2, 3]);
    assert_eq!(x.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        env.tensor_device("x"),
        "cpu",
        "round-trip must end on the cpu tag"
    );
}

#[test]
fn to_device_stamps_the_expected_tag() {
    let mut env = Environment::new();
    run(
        "x = reshape(iota(4), [2, 2])\n to_device(x, \"mlx\")",
        &mut env,
    );
    assert_eq!(env.tensor_device("x"), "mlx");
}

#[test]
fn to_device_rejects_unknown_target() {
    let mut env = Environment::new();
    // "cpu"/"mlx"/"cuda" are valid; anything else is rejected. Use a
    // clearly-bogus sentinel (not a real accelerator name like "tpu"/
    // "rocm" that could become a valid target later and silently turn
    // this rejection test into a no-op).
    let err = run_err(
        "x = iota(3)\n to_device(x, \"not-a-real-device\")",
        &mut env,
    );
    match err {
        EvalError::Unsupported(msg) => {
            assert!(
                msg.contains("not-a-real-device"),
                "error should name the bad target: {msg}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn to_device_accepts_cuda_and_stamps_tag() {
    // Saga cuda-foundation step 004: "cuda" is now a valid target.
    let mut env = Environment::new();
    run(
        "x = reshape(iota(4), [2, 2])\n to_device(x, \"cuda\")",
        &mut env,
    );
    assert_eq!(env.tensor_device("x"), "cuda");
}

// -- Device mismatch on apply --

#[test]
fn apply_errors_when_input_and_params_disagree_on_device() {
    // Model built outside any device block => params on cpu.
    // Input explicitly moved to mlx => apply should error.
    let mut env = Environment::new();
    let err = run_err(
        "m = linear(3, 4, 0)\n\
         x = reshape(iota(3), [1, 3])\n\
         to_device(x, \"mlx\")\n\
         apply(m, x)",
        &mut env,
    );
    match err {
        EvalError::DeviceMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, "apply");
            assert_eq!(expected, "cpu");
            assert_eq!(actual, "mlx");
        }
        other => panic!("expected DeviceMismatch, got {other:?}"),
    }
}

#[test]
fn apply_agrees_when_both_sides_on_mlx() {
    // Model built inside device("mlx") block -> params are mlx.
    // Input built and moved to mlx -> both sides agree.
    let mut env = Environment::new();
    let r = run(
        "device(\"mlx\") { m = linear(3, 4, 0) }\n\
         x = reshape(iota(3), [1, 3])\n\
         to_device(x, \"mlx\")\n\
         device(\"mlx\") { apply(m, x) }",
        &mut env,
    );
    assert_eq!(r.shape().dims(), &[1, 4]);
}

// -- Saga 11 model parity (gated on mlx feature + Apple Silicon) --

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod mlx_parity {
    use super::*;
    use mlpl_array::DenseArray;

    const FP32_TOL: f64 = 1e-3;

    fn assert_close(a: &DenseArray, b: &DenseArray, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch");
        assert_eq!(a.labels(), b.labels(), "label mismatch");
        for (i, (x, y)) in a.data().iter().zip(b.data().iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "elem {i}: cpu={x} mlx={y} diff={} tol={tol}",
                (x - y).abs()
            );
        }
    }

    fn run_forward(src: &str) -> DenseArray {
        let mut env = Environment::new();
        run(src, &mut env)
    }

    #[test]
    fn linear_forward_matches_cpu_within_fp32_tolerance() {
        let cpu = run_forward(
            "m = linear(4, 3, 7)\n\
             x = reshape(iota(8), [2, 4])\n\
             apply(m, x)",
        );
        let mlx = run_forward(
            "device(\"mlx\") { m = linear(4, 3, 7) }\n\
             x = reshape(iota(8), [2, 4])\n\
             to_device(x, \"mlx\")\n\
             device(\"mlx\") { apply(m, x) }",
        );
        assert_close(&cpu, &mlx, FP32_TOL);
    }

    #[test]
    fn chain_mlp_forward_matches_cpu_within_fp32_tolerance() {
        let cpu = run_forward(
            "m = chain(linear(4, 8, 1), tanh_layer(), linear(8, 2, 2))\n\
             x = reshape(iota(12), [3, 4])\n\
             apply(m, x)",
        );
        let mlx = run_forward(
            "device(\"mlx\") { m = chain(linear(4, 8, 1), tanh_layer(), linear(8, 2, 2)) }\n\
             x = reshape(iota(12), [3, 4])\n\
             to_device(x, \"mlx\")\n\
             device(\"mlx\") { apply(m, x) }",
        );
        assert_close(&cpu, &mlx, FP32_TOL);
    }

    #[test]
    fn causal_attention_forward_matches_cpu_within_fp32_tolerance() {
        // The Tiny LM's hottest block. Single-head, d_model=4, seq=4.
        let cpu = run_forward(
            "m = causal_attention(4, 1, 5)\n\
             x = reshape(iota(16), [4, 4])\n\
             apply(m, x)",
        );
        let mlx = run_forward(
            "device(\"mlx\") { m = causal_attention(4, 1, 5) }\n\
             x = reshape(iota(16), [4, 4])\n\
             to_device(x, \"mlx\")\n\
             device(\"mlx\") { apply(m, x) }",
        );
        assert_close(&cpu, &mlx, FP32_TOL);
    }

    #[test]
    fn tiny_lm_forward_matches_cpu_within_fp32_tolerance() {
        // Saga 13 Tiny LM-shaped forward: embed -> residual(rms_norm +
        // causal_attention) -> rms_norm -> linear. Scaled down (V=6,
        // d=4, T=4, heads=1) to keep the test cheap. Run it on both
        // devices with identical seeds; logits must agree within
        // fp32 tolerance.
        let model = "chain(embed(6, 4, 0), residual(chain(rms_norm(4), \
             causal_attention(4, 1, 1))), rms_norm(4), linear(4, 6, 2))";
        let cpu_src = format!("m = {model}\n x = [1, 3, 5, 2]\n apply(m, x)");
        let mlx_src = format!(
            "device(\"mlx\") {{ m = {model} }}\n \
             x = [1, 3, 5, 2]\n \
             to_device(x, \"mlx\")\n \
             device(\"mlx\") {{ apply(m, x) }}"
        );
        let cpu = run_forward(&cpu_src);
        let mlx = run_forward(&mlx_src);
        assert_close(&cpu, &mlx, FP32_TOL);
    }
}

// -- CUDA model parity (gated on cuda feature + Linux/x86_64) --
//
// Saga cuda-foundation step 004: inside `device("cuda") { }`,
// `apply(model, X)` routes matmul/softmax/add/etc through the CUDA
// runtime. Shapes + labels match the CPU path exactly; numbers match
// within the documented fp32 tolerance. Mirrors `mlx_parity`.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda_parity {
    use super::*;
    use mlpl_array::DenseArray;

    const FP32_TOL: f64 = 1e-3;

    fn assert_close(a: &DenseArray, b: &DenseArray, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch");
        assert_eq!(a.labels(), b.labels(), "label mismatch");
        for (i, (x, y)) in a.data().iter().zip(b.data().iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "elem {i}: cpu={x} cuda={y} diff={} tol={tol}",
                (x - y).abs()
            );
        }
    }

    fn run_forward(src: &str) -> DenseArray {
        let mut env = Environment::new();
        run(src, &mut env)
    }

    #[test]
    fn linear_forward_matches_cpu_within_fp32_tolerance() {
        let cpu = run_forward(
            "m = linear(4, 3, 7)\n\
             x = reshape(iota(8), [2, 4])\n\
             apply(m, x)",
        );
        let cuda = run_forward(
            "device(\"cuda\") { m = linear(4, 3, 7) }\n\
             x = reshape(iota(8), [2, 4])\n\
             to_device(x, \"cuda\")\n\
             device(\"cuda\") { apply(m, x) }",
        );
        assert_close(&cpu, &cuda, FP32_TOL);
    }

    #[test]
    fn chain_mlp_forward_matches_cpu_within_fp32_tolerance() {
        let cpu = run_forward(
            "m = chain(linear(4, 8, 1), tanh_layer(), linear(8, 2, 2))\n\
             x = reshape(iota(12), [3, 4])\n\
             apply(m, x)",
        );
        let cuda = run_forward(
            "device(\"cuda\") { m = chain(linear(4, 8, 1), tanh_layer(), linear(8, 2, 2)) }\n\
             x = reshape(iota(12), [3, 4])\n\
             to_device(x, \"cuda\")\n\
             device(\"cuda\") { apply(m, x) }",
        );
        assert_close(&cpu, &cuda, FP32_TOL);
    }

    #[test]
    fn tiny_lm_forward_matches_cpu_within_fp32_tolerance() {
        let model = "chain(embed(6, 4, 0), residual(chain(rms_norm(4), \
             causal_attention(4, 1, 1))), rms_norm(4), linear(4, 6, 2))";
        let cpu_src = format!("m = {model}\n x = [1, 3, 5, 2]\n apply(m, x)");
        let cuda_src = format!(
            "device(\"cuda\") {{ m = {model} }}\n \
             x = [1, 3, 5, 2]\n \
             to_device(x, \"cuda\")\n \
             device(\"cuda\") {{ apply(m, x) }}"
        );
        assert_close(&run_forward(&cpu_src), &run_forward(&cuda_src), FP32_TOL);
    }
}
