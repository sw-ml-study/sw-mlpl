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
