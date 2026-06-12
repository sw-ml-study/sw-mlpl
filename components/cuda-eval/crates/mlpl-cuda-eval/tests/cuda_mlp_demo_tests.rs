//! cuda-demo-parity step 2: validate the board-policy MLP fast path on
//! CUDA. The tic-tac-toe demo's model is `Chain[LinearLora, relu,
//! LinearLora]` (BOTH linears LoRA-adapted) -- a different architecture
//! than the head-only LoRA demo, with its own `eval_adam` fast path
//! (`grad_optim_cuda_mlp`). This runs the same fine-tune through the CPU
//! tape and the `device("cuda")` MLP path with identical seeds and
//! asserts the loss curves agree elementwise within fp32 tolerance AND
//! the GPU path actually trains (the curve moves). The CUDA analog of
//! the MLX MLP fast path; mirrors `cuda_lora_demo_tests.rs`.
//!
//! Triple-gated on Linux + x86_64 + the `cuda` feature, so
//! `cargo test --workspace` on other hosts skips this binary entirely.

#![cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

// fp32 forward+backward (candle) diverges a little each Adam step from
// the f64 CPU trajectory; the cross-path tolerance is fp32-realistic
// (matches the LoRA-head parity test). The point is "same learning".
const FP32_TOL: f64 = 5e-3;

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).expect("lex")).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

fn assert_close_slice(cpu: &[f64], cuda: &[f64], tol: f64) {
    assert_eq!(cpu.len(), cuda.len(), "length mismatch");
    for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
        assert!(
            (c - g).abs() <= tol,
            "elem {i}: cpu={c} cuda={g} diff={} tol={tol}",
            (c - g).abs()
        );
    }
}

// in=4, hidden=3, classes=2; 6 board-like rows -> integer move labels.
// `lora(chain(linear, relu_layer, linear))` produces the board-policy
// MLP shape (Chain[LinearLora, relu, LinearLora]) the fast path matches.
const SETUP: &str = "\
X = [[0.0, 0.2, -0.4, 0.4], [0.2, -0.2, 0.0, 0.4], [-0.4, 0.4, 0.2, 0.0], \
     [0.4, 0.0, 0.2, -0.2], [0.2, 0.4, -0.4, 0.0], [0.0, -0.4, 0.4, 0.2]]\n\
Y = [1, 0, 1, 0, 1, 0]\n\
base = chain(linear(4, 3, 1), relu_layer(), linear(3, 2, 2))\n\
student = lora(base, 2, 4.0, 7)\n\
";

fn run_finetune(cuda: bool) -> DenseArray {
    // S3: the CUDA step now lives in this crate; register it so
    // Environment::new picks it up for the device("cuda") fast path.
    mlpl_eval::register_gpu_step(mlpl_cuda_eval::gpu_step());
    let mut env = Environment::new();
    run(&mut env, SETUP);
    let finetune = "train 4 { \
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
fn mlp_cuda_finetune_matches_cpu_within_fp32_tolerance() {
    let cpu_losses = run_finetune(false);
    let cuda_losses = run_finetune(true);

    assert_eq!(cpu_losses.shape().dims(), &[4]);
    assert_eq!(cpu_losses.shape().dims(), cuda_losses.shape().dims());
    assert_close_slice(cpu_losses.data(), cuda_losses.data(), FP32_TOL);

    // Not two flat lines: the optimizer actually moved the loss on the
    // GPU, so a matching curve means matching *learning*.
    let g = cuda_losses.data();
    assert!(
        (g[g.len() - 1] - g[0]).abs() > 1e-6,
        "cuda mlp fine-tune should move the loss: {g:?}"
    );
}
