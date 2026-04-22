//! Saga 20 step 001: `clone_model(m) -> Model` builtin.
//!
//! `clone_model` deep-copies a `ModelSpec` tree and allocates
//! fresh param names so the caller can mutate the copy (via
//! `perturb_params`, `adam`, etc.) without touching the original.
//! These tests pin the contract:
//!
//! - Fresh, distinct param names on every clone.
//! - Pre-perturbation forward output matches the source bit-for-bit.
//! - Mutating the clone's params does not change the source.
//! - Nested clones produce distinct name sets from each other and
//!   from the source.
//! - New param names are registered as trainable.
//! - Device tags on the source's params propagate to the clone.
//! - Every Model DSL layer variant (linear, chain, residual, embed,
//!   rms_norm, causal_attention, activation) round-trips.

use std::collections::HashSet;

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

#[test]
fn clone_model_produces_fresh_param_names() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(4, 4, 0)");
    run(&mut env, "copy = clone_model(base)");

    let base_names = model_params(&env, "base").expect("base registered");
    let copy_names = model_params(&env, "copy").expect("copy registered");

    assert_eq!(base_names.len(), 2, "linear has W and b");
    assert_eq!(copy_names.len(), 2, "clone has W and b");
    assert_ne!(
        base_names, copy_names,
        "clone must not share param names with source"
    );

    let base_set: HashSet<_> = base_names.iter().collect();
    for name in &copy_names {
        assert!(
            !base_set.contains(name),
            "cloned param name '{name}' collides with a base param"
        );
    }
}

#[test]
fn clone_model_tensor_values_match_source() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(3, 2, 7)");
    run(&mut env, "copy = clone_model(base)");

    let base_names = model_params(&env, "base").unwrap();
    let copy_names = model_params(&env, "copy").unwrap();
    for (b, c) in base_names.iter().zip(copy_names.iter()) {
        let bv = env.get(b).expect("base param bound");
        let cv = env.get(c).expect("clone param bound");
        assert_eq!(bv.shape().dims(), cv.shape().dims());
        assert_eq!(
            bv.data(),
            cv.data(),
            "clone must copy the tensor values bit-for-bit"
        );
    }
}

#[test]
fn clone_model_forward_matches_source_before_mutation() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(2, 3, 1)");
    run(&mut env, "copy = clone_model(base)");
    env.set("X".into(), arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let base_out = run_expr(&mut env, "apply(base, X)");
    let copy_out = run_expr(&mut env, "apply(copy, X)");
    assert_eq!(base_out.shape().dims(), copy_out.shape().dims());
    assert_eq!(
        base_out.data(),
        copy_out.data(),
        "pre-mutation clone must be forward-identical to source"
    );
}

#[test]
fn mutating_clone_params_does_not_change_source() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(2, 2, 3)");
    run(&mut env, "copy = clone_model(base)");

    let base_names = model_params(&env, "base").unwrap();
    let copy_names = model_params(&env, "copy").unwrap();

    let base_w_before = env.get(&base_names[0]).unwrap().data().to_vec();

    // Overwrite the clone's W with a clearly-different tensor.
    env.set(
        copy_names[0].clone(),
        arr(vec![2, 2], vec![99.0, 99.0, 99.0, 99.0]),
    );

    let base_w_after = env.get(&base_names[0]).unwrap().data();
    assert_eq!(
        &base_w_before, base_w_after,
        "mutating the clone's W must not touch base's W"
    );
    let copy_w_after = env.get(&copy_names[0]).unwrap().data();
    assert_eq!(
        copy_w_after,
        &[99.0, 99.0, 99.0, 99.0],
        "clone's W must reflect the mutation"
    );
}

#[test]
fn nested_clones_have_distinct_names() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(2, 2, 0)");
    run(&mut env, "a = clone_model(base)");
    run(&mut env, "b = clone_model(a)");

    let base_names: HashSet<String> = model_params(&env, "base").unwrap().into_iter().collect();
    let a_names: HashSet<String> = model_params(&env, "a").unwrap().into_iter().collect();
    let b_names: HashSet<String> = model_params(&env, "b").unwrap().into_iter().collect();

    assert!(base_names.is_disjoint(&a_names), "a shares with base");
    assert!(base_names.is_disjoint(&b_names), "b shares with base");
    assert!(a_names.is_disjoint(&b_names), "b shares with a");
}

#[test]
fn clone_model_registers_new_params_as_trainable() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(3, 3, 5)");
    run(&mut env, "copy = clone_model(base)");

    let copy_names = model_params(&env, "copy").unwrap();
    for name in &copy_names {
        assert!(
            env.is_param(name),
            "cloned param '{name}' must be marked trainable"
        );
    }
}

#[test]
fn clone_model_walks_chain_residual_and_attention() {
    let mut env = Environment::new();
    let src = "base = chain(\
                 embed(16, 8, 0),\
                 residual(chain(rms_norm(8), causal_attention(8, 2, 1))),\
                 residual(chain(rms_norm(8), linear(8, 32, 2),\
                                relu_layer(), linear(32, 8, 3))),\
                 rms_norm(8),\
                 linear(8, 16, 4)\
               )";
    run(&mut env, src);
    run(&mut env, "copy = clone_model(base)");

    let base_names: HashSet<String> = model_params(&env, "base").unwrap().into_iter().collect();
    let copy_names: HashSet<String> = model_params(&env, "copy").unwrap().into_iter().collect();

    // Same parameter count; fully disjoint name sets.
    assert_eq!(base_names.len(), copy_names.len());
    assert!(
        base_names.is_disjoint(&copy_names),
        "deep clone must produce a fully disjoint name set"
    );

    // Spot-check that each family is represented in the clone's names.
    let has_attn = copy_names.iter().any(|n| n.starts_with("__attn_"));
    let has_embed = copy_names.iter().any(|n| n.starts_with("__embed_"));
    let has_linear = copy_names.iter().any(|n| n.starts_with("__linear_"));
    assert!(has_attn, "clone should own attention params");
    assert!(has_embed, "clone should own embedding params");
    assert!(has_linear, "clone should own linear params");
}

#[test]
fn clone_model_propagates_device_tags() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(2, 2, 0)");
    // Stamp base on MLX; clone must inherit the placement so
    // `apply(clone, X)` inside a `device("mlx") { }` block sees
    // the right device on every param.
    run(&mut env, "to_device(base, \"mlx\")");
    run(&mut env, "copy = clone_model(base)");

    let copy_names = model_params(&env, "copy").unwrap();
    for name in &copy_names {
        assert_eq!(
            env.tensor_device(name),
            "mlx",
            "cloned param '{name}' should inherit base's device tag"
        );
    }
}

#[test]
fn clone_model_requires_a_model_argument() {
    let mut env = Environment::new();
    run(&mut env, "x = 1");
    let stmts = parse(&lex("copy = clone_model(x)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("x is not a model");
    let msg = format!("{err:?}");
    assert!(
        msg.to_ascii_lowercase().contains("clone_model")
            || msg.to_ascii_lowercase().contains("model"),
        "error should reference clone_model or model, got: {msg}"
    );
}

#[test]
fn clone_model_rejects_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(2, 2, 0)");
    let stmts = parse(&lex("copy = clone_model(base, base)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity mismatch");
    let msg = format!("{err:?}");
    assert!(
        msg.to_ascii_lowercase().contains("clone_model")
            || msg.to_ascii_lowercase().contains("arity"),
        "error should reference clone_model or arity, got: {msg}"
    );
}
