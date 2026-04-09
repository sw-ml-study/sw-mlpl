//! Saga 11 step 001: Value::Model + linear() atomic layer.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn linear_creates_model_with_correct_param_shapes() {
    let mut env = Environment::new();
    let stmts = parse(&lex("L = linear(2, 3, 7)").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();
    let names = model_params(&env, "L").expect("L is a registered model");
    assert_eq!(names.len(), 2, "linear has W and b");
    let w = env.get(&names[0]).expect("W bound");
    let b = env.get(&names[1]).expect("b bound");
    assert_eq!(w.shape().dims(), &[2, 3]);
    assert_eq!(b.shape().dims(), &[1, 3]);
    // Both must be tracked as trainable params.
    assert!(env.is_param(&names[0]));
    assert!(env.is_param(&names[1]));
}

#[test]
fn apply_linear_runs_matmul_plus_bias() {
    // Build a linear layer manually so we know the exact W and b.
    let mut env = Environment::new();
    let stmts = parse(&lex("L = linear(2, 3, 1)").unwrap()).unwrap();
    eval_program(&stmts, &mut env).unwrap();

    // Overwrite W and b with deterministic values.
    let names = model_params(&env, "L").unwrap();
    env.set(
        names[0].clone(),
        arr(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    );
    env.set(names[1].clone(), arr(vec![1, 3], vec![10.0, 20.0, 30.0]));

    // Apply on a [2, 2] input.
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let stmts = parse(&lex("apply(L, X)").unwrap()).unwrap();
    let out = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(out.shape().dims(), &[2, 3]);
    // Row 0: [1, 2] * W = [1, 2, 0]; +b = [11, 22, 30]
    // Row 1: [3, 4] * W = [3, 4, 0]; +b = [13, 24, 30]
    assert_eq!(out.data(), &[11.0, 22.0, 30.0, 13.0, 24.0, 30.0]);
}

#[test]
fn chain_mlp_applies_through_layers_and_normalizes() {
    // 2 -> 8 -> 2 chain ending in softmax: rows must sum to 1.
    let mut env = Environment::new();
    let src = "M = chain(linear(2, 8, 1), tanh_layer(), linear(8, 2, 2), softmax_layer())";
    eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env).unwrap();
    let names = model_params(&env, "M").unwrap();
    assert_eq!(names.len(), 4, "two linear layers contribute W and b each");

    env.set(
        "X".into(),
        arr(vec![3, 2], vec![0.5, -0.5, 1.0, 0.0, -1.0, 1.0]),
    );
    let stmts = parse(&lex("apply(M, X)").unwrap()).unwrap();
    let out = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(out.shape().dims(), &[3, 2]);
    for row in 0..3 {
        let s = out.data()[row * 2] + out.data()[row * 2 + 1];
        assert!(
            (s - 1.0).abs() < 1e-9,
            "row {row} of softmax should sum to 1, got {s}"
        );
        assert!(out.data()[row * 2] > 0.0);
        assert!(out.data()[row * 2 + 1] > 0.0);
    }
}

#[test]
fn relu_layer_clamps_negatives() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("R = chain(relu_layer())").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    env.set("X".into(), arr(vec![1, 4], vec![-2.0, -0.5, 0.0, 3.0]));
    let out = eval_program(&parse(&lex("apply(R, X)").unwrap()).unwrap(), &mut env).unwrap();
    assert_eq!(out.data(), &[0.0, 0.0, 0.0, 3.0]);
}

#[test]
fn adam_walks_model_params_and_drives_loss_to_zero() {
    // chain[linear(2, 2, 1)] -> two params (__linear_W_0, __linear_b_0).
    // Loss = sum(W*W) + sum(b*b). adam(loss, M, ...) should walk M's
    // param tree and drive both params toward zero.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("M = chain(linear(2, 2, 1))").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "M").unwrap();
    assert_eq!(names.len(), 2);
    // Force W to a known non-zero starting point so the test isn't
    // sensitive to the random init.
    env.set(names[0].clone(), arr(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]));
    env.set(names[1].clone(), arr(vec![1, 2], vec![0.5, -0.5]));

    let src = format!(
        "train 100 {{ adam(sum({w}*{w}) + sum({b}*{b}), M, 0.05, 0.9, 0.999, 0.00000001) }}",
        w = names[0],
        b = names[1],
    );
    eval_program(&parse(&lex(&src).unwrap()).unwrap(), &mut env).unwrap();
    let w = env.get(&names[0]).unwrap();
    let b = env.get(&names[1]).unwrap();
    let w_norm: f64 = w.data().iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm: f64 = b.data().iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(w_norm < 0.1, "expected W ~= 0, got norm {w_norm}");
    assert!(b_norm < 0.1, "expected b ~= 0, got norm {b_norm}");
}

#[test]
fn momentum_sgd_also_walks_model_params() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("M = chain(linear(1, 1, 3))").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "M").unwrap();
    env.set(names[0].clone(), arr(vec![1, 1], vec![2.0]));
    env.set(names[1].clone(), arr(vec![1, 1], vec![1.0]));

    let src = format!(
        "train 100 {{ momentum_sgd(sum({w}*{w}) + sum({b}*{b}), M, 0.05, 0.9) }}",
        w = names[0],
        b = names[1],
    );
    eval_program(&parse(&lex(&src).unwrap()).unwrap(), &mut env).unwrap();
    let w = env.get(&names[0]).unwrap().data()[0];
    let b = env.get(&names[1]).unwrap().data()[0];
    assert!(w.abs() < 0.1, "expected W ~= 0, got {w}");
    assert!(b.abs() < 0.1, "expected b ~= 0, got {b}");
}

#[test]
fn residual_with_zero_inner_block_returns_input_unchanged() {
    // Build R = residual(chain(linear(2, 2, 9))) and force the inner
    // linear's W and b to zero so the inner output is the zero matrix.
    // Then R(X) = X + 0 = X.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("R = residual(chain(linear(2, 2, 9)))").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "R").unwrap();
    assert_eq!(
        names.len(),
        2,
        "residual recurses into the linear's W and b"
    );
    env.set(names[0].clone(), arr(vec![2, 2], vec![0.0; 4]));
    env.set(names[1].clone(), arr(vec![1, 2], vec![0.0; 2]));

    let xv = vec![0.5, -0.5, 1.0, 0.0, -1.0, 1.0];
    env.set("X".into(), arr(vec![3, 2], xv.clone()));
    let out = eval_program(&parse(&lex("apply(R, X)").unwrap()).unwrap(), &mut env).unwrap();
    assert_eq!(out.shape().dims(), &[3, 2]);
    assert_eq!(out.data(), xv.as_slice());
}

#[test]
fn rms_norm_produces_unit_rms_per_row() {
    let mut env = Environment::new();
    eval_program(&parse(&lex("N = rms_norm(4)").unwrap()).unwrap(), &mut env).unwrap();
    assert!(
        model_params(&env, "N").unwrap().is_empty(),
        "rms_norm has no params"
    );

    env.set(
        "X".into(),
        arr(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 0.5, 1.5]),
    );
    let out = eval_program(&parse(&lex("apply(N, X)").unwrap()).unwrap(), &mut env).unwrap();
    assert_eq!(out.shape().dims(), &[2, 4]);
    for r in 0..2 {
        let row = &out.data()[r * 4..(r + 1) * 4];
        let mean_sq: f64 = row.iter().map(|v| v * v).sum::<f64>() / 4.0;
        let rms = mean_sq.sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-6,
            "row {r} expected unit RMS, got {rms}"
        );
    }
}

#[test]
fn linear_with_same_seed_produces_same_initial_weights() {
    let mut env_a = Environment::new();
    let mut env_b = Environment::new();
    let stmts = parse(&lex("L = linear(2, 4, 42)").unwrap()).unwrap();
    eval_program(&stmts, &mut env_a).unwrap();
    eval_program(&stmts, &mut env_b).unwrap();
    let na = model_params(&env_a, "L").unwrap();
    let nb = model_params(&env_b, "L").unwrap();
    assert_eq!(
        env_a.get(&na[0]).unwrap().data(),
        env_b.get(&nb[0]).unwrap().data()
    );
}
