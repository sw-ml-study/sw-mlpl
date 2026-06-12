//! Saga cuda-foundation step 006: validate `demos/lora_finetune_cuda.mlpl`.
//!
//! Runs a cut-down LoRA fine-tune through both the CPU and the CUDA
//! paths (fine-tune train wrapped in `device("cuda")`) with identical
//! seeds, and asserts the fine-tune loss curves agree elementwise
//! within fp32 tolerance AND the CUDA path actually learns (the loss
//! drops). The CUDA analog of `lora_mlx_demo_tests.rs`.
//!
//! Triple-gated on Linux + x86_64 + the `cuda` feature, so
//! `cargo test --workspace` on other hosts skips this binary entirely.

#![cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

// device("cuda") runs the forward AND backward in fp32 via candle
// autograd; genuine fp32 gradients diverge from the f64 CPU trajectory
// a little each Adam step, so the cross-path tolerance is fp32-realistic
// (matches the MLX parity test). The curves still track closely; the
// point is "same learning", not bit-identity with f64.
const FP32_TOL: f64 = 5e-3;

fn run(env: &mut Environment, src: &str) {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

fn assert_close_slice(cpu: &[f64], cuda: &[f64], tol: f64, label: &str) {
    assert_eq!(cpu.len(), cuda.len(), "{label}: length mismatch");
    for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
        assert!(
            (c - g).abs() <= tol,
            "{label} elem {i}: cpu={c} cuda={g} diff={} tol={tol}",
            (c - g).abs()
        );
    }
}

// Seed-aligned base + corpus used by both runs (V=16, d=8, h=1, rank=2,
// 3 base + 3 fine-tune steps). Matches the MLX parity test's setup.
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

fn run_finetune(cuda: bool) -> DenseArray {
    // S3: the CUDA step now lives in this crate; register it so
    // Environment::new picks it up for the device("cuda") fast path.
    mlpl_eval::register_gpu_step(mlpl_cuda_eval::gpu_step());
    let mut env = Environment::new();
    run(&mut env, BASE_SETUP);
    let finetune = "train 3 { \
        adam(cross_entropy(apply(student, X), Y), student, \
             0.05, 0.9, 0.999, 0.00000001); \
        loss_metric = cross_entropy(apply(student, X), Y) \
    }";
    if cuda {
        let wrapped = format!(
            "device(\"cuda\") {{ \
               to_device(student, \"cuda\"); \
               to_device(X, \"cuda\"); \
               {finetune} \
             }}"
        );
        run(&mut env, &wrapped);
    } else {
        run(&mut env, finetune);
    }
    env.get("last_losses").expect("last_losses bound").clone()
}

#[test]
fn lora_cuda_finetune_matches_cpu_within_fp32_tolerance() {
    let cpu_losses = run_finetune(false);
    let cuda_losses = run_finetune(true);

    assert_eq!(cpu_losses.shape().dims(), &[3]);
    assert_eq!(cpu_losses.shape().dims(), cuda_losses.shape().dims());
    assert_close_slice(
        cpu_losses.data(),
        cuda_losses.data(),
        FP32_TOL,
        "last_losses",
    );

    // The CUDA path actually learns (the recorded curve drops), so a
    // matching curve means matching learning, not two flat lines.
    let gl = cuda_losses.data();
    assert!(
        gl[2] < gl[0],
        "cuda fine-tune should reduce the loss: {gl:?}"
    );
}
