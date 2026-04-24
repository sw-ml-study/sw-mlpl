//! Saga 22 step 001: `estimate_train(...)` feasibility /
//! resource-estimation builtin.
//!
//! Pure-math estimator over `ModelSpec` + loop shape.
//! Returns a rank-1 `[5]` f64 array with
//! `[params, vram_bytes, disk_bytes, flops,
//! wall_seconds]`. See
//! `contracts/eval-contract/estimate.md`.

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn estimate(env: &mut Environment, src: &str) -> Vec<f64> {
    let out = run(env, src);
    assert_eq!(
        out.shape().dims(),
        &[5],
        "estimate_train should return rank-1 [5]"
    );
    out.data().to_vec()
}

/// Tolerate floating-point drift on wall_seconds (division
/// by a f64 throughput). All other components are exact.
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9_f64.max(b.abs() * 1e-12)
}

#[test]
fn estimate_tiny_linear_exact() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    let est = estimate(&mut env, "estimate_train(m, 100, 32, 1)");
    // params = |W|=12 + |b|=4 = 16
    assert_eq!(est[0], 16.0);
    // trainable = 16 (no freeze)
    // vram = (params + trainable + 2*trainable) * 8
    //      + batch * seq * hidden * depth * 8 * 4
    //      = (16 + 16 + 32)*8 + 32*1*4*1*8*4
    //      = 512 + 4096 = 4608
    assert_eq!(est[1], 4608.0);
    // disk = 16 * 8 = 128
    assert_eq!(est[2], 128.0);
    // flops = 2*3*4*32*100 = 76800
    assert_eq!(est[3], 76800.0);
    // wall = 76800 / 50e9 (default throughput)
    assert!(
        approx_eq(est[4], 76800.0 / 50e9),
        "wall = {} expected {}",
        est[4],
        76800.0 / 50e9
    );
}

#[test]
fn estimate_chain_is_additive() {
    let mut env = Environment::new();
    run(&mut env, "m = chain(linear(3, 4, 0), linear(4, 5, 1))");
    let est = estimate(&mut env, "estimate_train(m, 10, 8, 1)");
    // params = (12+4) + (20+5) = 41
    assert_eq!(est[0], 41.0);
    // depth = 2, hidden = max(3,4,5) = 5
    // vram = (41 + 41 + 82)*8 + 8*1*5*2*8*4 = 1312 + 2560 = 3872
    assert_eq!(est[1], 3872.0);
    // disk = 41 * 8 = 328
    assert_eq!(est[2], 328.0);
    // flops = (2*3*4*8 + 2*4*5*8) * 10 = (192 + 320) * 10 = 5120
    assert_eq!(est[3], 5120.0);
}

#[test]
fn estimate_lora_freezes_base_from_vram_but_counts_all_params() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(10, 10, 0)");
    run(&mut env, "student = lora(base, 4, 16.0, 7)");
    let est = estimate(&mut env, "estimate_train(student, 10, 4, 1)");
    // Full params: |W|=100 + |b|=10 + |A|=40 + |B_adapter|=40 = 190
    assert_eq!(est[0], 190.0);
    // trainable = A + B_adapter = 80 (base W, b frozen by lora)
    // vram = (190 + 80 + 160)*8 + 4*1*10*1*8*4 = 3440 + 1280 = 4720
    assert_eq!(est[1], 4720.0);
    // disk counts every param including frozen base.
    assert_eq!(est[2], 190.0 * 8.0);
    // flops per step: main 2*10*10*4 + adapters 2*10*4*4 + 2*4*10*4
    //               = 800 + 320 + 320 = 1440
    // flops total = 14400
    assert_eq!(est[3], 14400.0);
}

#[test]
fn estimate_embedding_scales_with_vocab() {
    let mut env = Environment::new();
    run(&mut env, "e_small = embed(100, 16, 0)");
    run(&mut env, "e_big = embed(1000, 16, 1)");
    let small = estimate(&mut env, "estimate_train(e_small, 10, 8, 4)");
    let big = estimate(&mut env, "estimate_train(e_big, 10, 8, 4)");
    // params scale 10x
    assert_eq!(small[0], 1600.0);
    assert_eq!(big[0], 16000.0);
    // flops scale 10x (2 * batch * vocab * d_model * steps)
    assert!(
        approx_eq(big[3] / small[3], 10.0),
        "flops ratio {} should be 10",
        big[3] / small[3]
    );
}

#[test]
fn estimate_attention_adds_activation_and_flops() {
    let mut env = Environment::new();
    run(
        &mut env,
        "plain = chain(embed(64, 16, 0), linear(16, 64, 1))",
    );
    run(
        &mut env,
        "attn = chain(embed(64, 16, 0), causal_attention(16, 2, 2), linear(16, 64, 1))",
    );
    let p = estimate(&mut env, "estimate_train(plain, 1, 8, 4)");
    let a = estimate(&mut env, "estimate_train(attn, 1, 8, 4)");
    // Attention adds flops (projections + seq^2 matmuls).
    assert!(
        a[3] > p[3],
        "attention flops {} must exceed plain {}",
        a[3],
        p[3]
    );
    // Attention adds params (Wq, Wk, Wv, Wo each 16x16 = 256; 4*256 = 1024).
    assert!(
        a[0] >= p[0] + 1024.0,
        "attention params {} must exceed plain {} + 1024",
        a[0],
        p[0]
    );
}

#[test]
fn estimate_dtype_bytes_halves_memory() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(8, 16, 0)");
    let def = estimate(&mut env, "estimate_train(m, 100, 32, 1)");
    let f32 = estimate(&mut env, "estimate_train(m, 100, 32, 1, 4)");
    // disk exactly halves
    assert!(approx_eq(f32[2], def[2] / 2.0));
    // vram exactly halves (both weight and activation components scale with dtype)
    assert!(approx_eq(f32[1], def[1] / 2.0));
    // flops do not change
    assert_eq!(f32[3], def[3]);
}

#[test]
fn estimate_throughput_override_scales_wall() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    let default_wall = estimate(&mut env, "estimate_train(m, 100, 32, 1)")[4];
    env.set_string("mlpl_device_throughput_gflops".into(), "500.0".into());
    let fast_wall = estimate(&mut env, "estimate_train(m, 100, 32, 1)")[4];
    assert!(
        approx_eq(default_wall / fast_wall, 10.0),
        "500 GFLOPS should give 10x less wall-clock than default 50"
    );
}

#[test]
fn estimate_errors_on_non_model_arg() {
    let mut env = Environment::new();
    env.set("X".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    let stmts = parse(&lex("estimate_train(X, 10, 4, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("non-model should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_train") || msg.contains("model"),
        "got: {msg}"
    );
}

#[test]
fn estimate_errors_on_zero_batch() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    let stmts = parse(&lex("estimate_train(m, 10, 0, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("zero batch should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_train") && (msg.contains("positive") || msg.contains("batch")),
        "got: {msg}"
    );
}

#[test]
fn estimate_errors_on_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(3, 4, 0)");
    let stmts = parse(&lex("estimate_train(m, 10)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_train") || msg.contains("arity"),
        "got: {msg}"
    );
}

#[test]
fn estimate_errors_on_activation_only_model() {
    let mut env = Environment::new();
    run(&mut env, "m = chain(relu_layer(), tanh_layer())");
    let stmts = parse(&lex("estimate_train(m, 10, 4, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("no params should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("estimate_train") && msg.contains("no trainable"),
        "got: {msg}"
    );
}
