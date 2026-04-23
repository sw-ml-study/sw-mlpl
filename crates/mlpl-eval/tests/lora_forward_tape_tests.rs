//! Saga 15 step 003: forward + autograd for `LinearLora`.
//!
//! Pins three invariants:
//! 1. Before any training, `apply(lora_m, X) == apply(linear_m, X)`
//!    because B is zero-initialized.
//! 2. The numerical forward matches `y = X @ W + (alpha/rank) *
//!    X @ A @ B + b` when W, A, B, b are hand-constructed.
//! 3. `adam(loss, lora_m, ...)` with the auto-frozen base
//!    leaves W, b bit-identical across N steps while A, B
//!    move (Saga 15 step 001's frozen_params plus step 002's
//!    auto-freeze, verified through the full forward-and-
//!    backward path).

use std::collections::HashMap;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

fn run_expr(env: &mut Environment, src: &str) -> DenseArray {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap()
}

fn snapshot_params(env: &Environment, model_ident: &str) -> HashMap<String, Vec<f64>> {
    model_params(env, model_ident)
        .unwrap()
        .into_iter()
        .map(|n| {
            let v = env.get(&n).unwrap().data().to_vec();
            (n, v)
        })
        .collect()
}

#[test]
fn forward_identity_before_training() {
    // Zero-init B means the adapter delta is zero at rest,
    // so `apply(lora_m, X)` matches the cloned base's
    // forward elementwise.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 3, 1)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    env.set(
        "X".into(),
        arr(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );

    // The base's clone inside `student` has fresh names; build
    // a fresh Linear with the same underlying W, b so we can
    // compare forwards.
    let student_names = model_params(&env, "student").unwrap();
    let student_w = student_names
        .iter()
        .find(|n| n.starts_with("__linear_W_"))
        .unwrap()
        .clone();
    let student_b = student_names
        .iter()
        .find(|n| n.starts_with("__linear_b_"))
        .unwrap()
        .clone();

    // Reconstruct the base forward manually using the cloned W, b.
    let x = env.get("X").unwrap().clone();
    let w = env.get(&student_w).unwrap().clone();
    let b = env.get(&student_b).unwrap().clone();
    let xw = x.matmul(&w).unwrap();
    let ones = arr(vec![2, 1], vec![1.0; 2]);
    let b_broadcast = ones.matmul(&b).unwrap();
    let expected = xw.apply_binop(&b_broadcast, |a, b| a + b).unwrap();

    let lora_out = run_expr(&mut env, "apply(student, X)");
    assert_eq!(lora_out.shape().dims(), expected.shape().dims());
    for (i, (l, e)) in lora_out.data().iter().zip(expected.data()).enumerate() {
        assert!(
            (l - e).abs() < 1e-12,
            "forward identity violated at {i}: lora={l}, base={e}"
        );
    }
}

#[test]
fn forward_matches_hand_constructed_formula() {
    // Hand-pick W = I, b = 0, A and B of known values so the
    // expected output is an exact integer lattice.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 4, 0)");
    run(&mut env, "student = lora(m, 2, 2.0, 7)"); // alpha/rank = 1

    // Overwrite base and adapter params with deterministic values.
    let names = model_params(&env, "student").unwrap();
    let w_name = names.iter().find(|n| n.starts_with("__linear_W_")).unwrap();
    let b_name = names.iter().find(|n| n.starts_with("__linear_b_")).unwrap();
    let a_name = names.iter().find(|n| n.starts_with("__lora_A_")).unwrap();
    let b_adapter_name = names.iter().find(|n| n.starts_with("__lora_B_")).unwrap();

    env.set(
        w_name.clone(),
        arr(
            vec![4, 4],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
    );
    env.set(b_name.clone(), arr(vec![1, 4], vec![0.0; 4]));
    env.set(
        a_name.clone(),
        arr(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    env.set(
        b_adapter_name.clone(),
        arr(vec![2, 4], vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]),
    );

    env.set("X".into(), arr(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]));
    let out = run_expr(&mut env, "apply(student, X)");

    // Expected: X @ W + 1.0 * (X @ A @ B) + b
    //         = [1,0,0,0] + [1,0,0,0]@[[1,0],[0,1],[0,0],[0,0]]@[[2,0,0,0],[0,3,0,0]]
    //         = [1,0,0,0] + [1,0]@[[2,0,0,0],[0,3,0,0]]
    //         = [1,0,0,0] + [2,0,0,0]
    //         = [3,0,0,0]
    assert_eq!(out.shape().dims(), &[1, 4]);
    assert_eq!(out.data(), &[3.0, 0.0, 0.0, 0.0]);
}

#[test]
fn forward_scales_adapter_by_alpha_over_rank() {
    // Same setup as above but alpha=4, rank=2 so scale=2.
    // The adapter contribution doubles.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 4, 0)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)"); // alpha/rank = 2

    let names = model_params(&env, "student").unwrap();
    let w_name = names.iter().find(|n| n.starts_with("__linear_W_")).unwrap();
    let b_name = names.iter().find(|n| n.starts_with("__linear_b_")).unwrap();
    let a_name = names.iter().find(|n| n.starts_with("__lora_A_")).unwrap();
    let b_adapter_name = names.iter().find(|n| n.starts_with("__lora_B_")).unwrap();

    env.set(
        w_name.clone(),
        arr(
            vec![4, 4],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
    );
    env.set(b_name.clone(), arr(vec![1, 4], vec![0.0; 4]));
    env.set(
        a_name.clone(),
        arr(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    env.set(
        b_adapter_name.clone(),
        arr(vec![2, 4], vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]),
    );

    env.set("X".into(), arr(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]));
    let out = run_expr(&mut env, "apply(student, X)");

    // Expected: [1,0,0,0] + 2 * [2,0,0,0] = [5, 0, 0, 0]
    assert_eq!(out.data(), &[5.0, 0.0, 0.0, 0.0]);
}

#[test]
fn adam_leaves_frozen_base_untouched_and_moves_adapters() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 4, 1)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    env.set(
        "X".into(),
        arr(
            vec![3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ),
    );
    env.set(
        "Y".into(),
        arr(
            vec![3, 4],
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
    );

    let before = snapshot_params(&env, "student");
    run(
        &mut env,
        "train 3 { adam(mean((apply(student, X) - Y) * (apply(student, X) - Y)), student, 0.05, 0.9, 0.999, 0.00000001); loss_metric = mean((apply(student, X) - Y) * (apply(student, X) - Y)) }",
    );
    let after = snapshot_params(&env, "student");

    for (name, before_vals) in &before {
        let after_vals = after.get(name).unwrap();
        let is_base_w = name.starts_with("__linear_W_");
        let is_base_b = name.starts_with("__linear_b_");
        let is_adapter_a = name.starts_with("__lora_A_");
        let is_adapter_b = name.starts_with("__lora_B_");

        if is_base_w || is_base_b {
            assert_eq!(
                before_vals, after_vals,
                "frozen base param '{name}' must be bit-identical after training"
            );
        }
        if is_adapter_a || is_adapter_b {
            assert_ne!(
                before_vals, after_vals,
                "adapter '{name}' should have moved after 3 adam steps"
            );
        }
    }
}

#[test]
fn unfreeze_enables_training_on_the_full_lora_model() {
    // Saga 15 contract: `unfreeze(student)` after `lora()`
    // opens the base W, b to training alongside the adapters.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 4, 2)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    run(&mut env, "unfreeze(student)");
    env.set(
        "X".into(),
        arr(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );
    env.set("Y".into(), arr(vec![2, 4], vec![0.5; 8]));

    let before = snapshot_params(&env, "student");
    run(
        &mut env,
        "train 3 { adam(mean((apply(student, X) - Y) * (apply(student, X) - Y)), student, 0.05, 0.9, 0.999, 0.00000001); loss_metric = mean((apply(student, X) - Y) * (apply(student, X) - Y)) }",
    );
    let after = snapshot_params(&env, "student");

    // Now EVERY param should have moved (including W, b).
    for (name, before_vals) in &before {
        let after_vals = after.get(name).unwrap();
        assert_ne!(
            before_vals, after_vals,
            "after unfreeze + training, every param should move (incl. '{name}')"
        );
    }
}

#[test]
fn lora_forward_compose_in_chain() {
    // Compose LoRA with the rest of the Model DSL: `lora` on
    // a chain should wrap both linears and the chain's
    // forward should still work end-to-end.
    let mut env = Environment::new();
    run(
        &mut env,
        "m = chain(linear(4, 8, 0), relu_layer(), linear(8, 4, 1))",
    );
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    env.set("X".into(), arr(vec![2, 4], vec![1.0; 8]));

    // Before training: should match the base's forward.
    let student_out = run_expr(&mut env, "apply(student, X)");
    assert_eq!(student_out.shape().dims(), &[2, 4]);
    for v in student_out.data() {
        assert!(v.is_finite(), "every output should be finite");
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
#[test]
fn lora_forward_matches_cpu_on_mlx() {
    // Triple-gated MLX parity: CPU path and MLX path produce
    // element-agreeing forwards within fp32 tolerance.
    const TOL: f64 = 1e-3;

    // CPU run.
    let mut cpu = Environment::new();
    run(&mut cpu, "m = linear(4, 4, 3)");
    run(&mut cpu, "student = lora(m, 2, 4.0, 5)");
    cpu.set(
        "X".into(),
        arr(vec![2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    );
    let cpu_out = run_expr(&mut cpu, "apply(student, X)");

    // MLX run with identical seeds + identical X.
    let mut mlx = Environment::new();
    run(&mut mlx, "device(\"mlx\") { m = linear(4, 4, 3) }");
    run(&mut mlx, "device(\"mlx\") { student = lora(m, 2, 4.0, 5) }");
    mlx.set(
        "X".into(),
        arr(vec![2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    );
    run(&mut mlx, "to_device(X, \"mlx\")");
    let mlx_out = run_expr(&mut mlx, "device(\"mlx\") { apply(student, X) }");

    assert_eq!(cpu_out.shape().dims(), mlx_out.shape().dims());
    for (i, (c, m)) in cpu_out.data().iter().zip(mlx_out.data()).enumerate() {
        assert!(
            (c - m).abs() <= TOL,
            "CPU/MLX parity: elem {i}: cpu={c} mlx={m}"
        );
    }
}
