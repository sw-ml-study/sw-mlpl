//! Saga 14 step 006: gradcheck parity for `grad(expr, wrt)` inside
//! `device("mlx") { }`.
//!
//! For each tape primitive, build the same scalar-output expression
//! both ways -- raw `grad(...)` (CPU forward + CPU backward) and
//! `device("mlx") { grad(...) }` (MLX-rounded forward values + the
//! same CPU backward formulas) -- and assert the resulting gradient
//! matches within fp32 tolerance.
//!
//! The MLX path differs from CPU only in that each tape node's
//! forward value is re-materialized through `mlpl-mlx-rt` after the
//! CPU forward built the tape (see
//! `crate::device::materialize_tape_on_mlx` for the rationale).
//! Backward formulas operate on those MLX-rounded values, which
//! is the only source of divergence the tolerance budget needs to
//! absorb.
//!
//! Triple-gated like the rest of `mlpl-mlx-rt`.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const FP32_TOL: f64 = 1e-4;

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

/// Run the same `grad(...)` expression on CPU and inside
/// `device("mlx") { ... }` and assert the gradients agree.
fn check_grad_parity(setup: &str, grad_expr: &str, label: &str) {
    let cpu_src = format!("{setup}\n {grad_expr}");
    let mlx_src = format!("{setup}\n device(\"mlx\") {{ {grad_expr} }}");
    let cpu = run(&cpu_src);
    let mlx = run(&mlx_src);
    assert_close(&cpu, &mlx, FP32_TOL, label);
}

// ---- per-primitive parity ----

#[test]
fn grad_sum_parity() {
    check_grad_parity("w = param[3]", "grad(sum(w), w)", "sum");
}

#[test]
fn grad_mean_parity() {
    check_grad_parity("w = param[4]", "grad(mean(w), w)", "mean");
}

#[test]
fn grad_exp_parity() {
    check_grad_parity("w = param[3]", "grad(sum(exp(w)), w)", "exp");
}

#[test]
fn grad_log_parity() {
    // Log inputs must be > 0 to avoid -inf; param[3] init can be
    // negative, so add a positive scalar offset before log.
    check_grad_parity("w = param[3]", "grad(sum(log(w + 2.0)), w)", "log");
}

#[test]
fn grad_relu_parity() {
    check_grad_parity("w = param[3]", "grad(sum(relu(w)), w)", "relu");
}

#[test]
fn grad_tanh_parity() {
    check_grad_parity("w = param[3]", "grad(sum(tanh(w)), w)", "tanh");
}

#[test]
fn grad_sigmoid_parity() {
    check_grad_parity("w = param[3]", "grad(sum(sigmoid(w)), w)", "sigmoid");
}

#[test]
fn grad_neg_parity() {
    check_grad_parity("w = param[3]", "grad(sum(-w), w)", "neg");
}

#[test]
fn grad_add_parity() {
    check_grad_parity(
        "w = param[3]\n c = [0.5, 1.0, -0.5]",
        "grad(sum(w + c), w)",
        "add",
    );
}

#[test]
fn grad_mul_parity() {
    check_grad_parity(
        "w = param[3]\n c = [0.5, 1.0, -0.5]",
        "grad(sum(w * c), w)",
        "mul",
    );
}

#[test]
fn grad_div_parity() {
    check_grad_parity(
        "w = param[3]\n c = [2.0, 3.0, 4.0]",
        "grad(sum(w / c), w)",
        "div",
    );
}

#[test]
fn grad_softmax_parity() {
    check_grad_parity("w = param[4]", "grad(sum(softmax(w)), w)", "softmax");
}

#[test]
fn grad_transpose_parity() {
    check_grad_parity("w = param[2, 3]", "grad(sum(transpose(w)), w)", "transpose");
}

#[test]
fn grad_matmul_parity() {
    check_grad_parity(
        "w = param[3, 2]\n x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]",
        "grad(sum(matmul(x, w)), w)",
        "matmul",
    );
}

// ---- composition + chain rule ----

#[test]
fn grad_chained_ops_parity() {
    // A few ops stacked: tanh(matmul(x, w)) -> mean.
    // Parity must hold across the chain rule, not just one op.
    check_grad_parity(
        "w = param[3, 2]\n x = [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]",
        "grad(mean(tanh(matmul(x, w))), w)",
        "chained",
    );
}

// ---- integration: Tiny LM-shaped slice ----

#[test]
fn grad_cross_entropy_apply_linear_parity() {
    // grad of cross_entropy(apply(linear, X), targets) wrt the
    // linear's W is the closing test for step 006.
    let setup = "m = linear(4, 3, 0)\n \
                 X = reshape(iota(8), [2, 4])\n \
                 Y = [0, 2]";
    let grad_expr = "grad(cross_entropy(apply(m, X), Y), __linear_W_0)";
    let cpu = run(&format!("{setup}\n {grad_expr}"));
    // For MLX, the model must be built inside the mlx scope so its
    // params are tagged "mlx", and X must be moved before apply.
    let mlx_src = "device(\"mlx\") { m = linear(4, 3, 0) }\n \
                   X = reshape(iota(8), [2, 4])\n \
                   to_device(X, \"mlx\")\n \
                   Y = [0, 2]\n \
                   device(\"mlx\") { grad(cross_entropy(apply(m, X), Y), __linear_W_0) }";
    let mlx = run(mlx_src);
    assert_close(&cpu, &mlx, FP32_TOL, "cross_entropy(apply(linear))");
}
