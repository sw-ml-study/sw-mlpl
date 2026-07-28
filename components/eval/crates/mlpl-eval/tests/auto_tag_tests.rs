//! Saga 23 step 002: auto-tagging from producer ops.
//!
//! Each test runs a small MLPL program that exercises one producer
//! rule and asserts the resulting binding carries the expected
//! `ValueTag` in `Environment::tags`.

use mlpl_array::DenseArray;
use mlpl_core::{LossKind, ValueTag};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

#[test]
fn softmax_tags_result_as_probability() {
    let mut env = Environment::new();
    run(
        "logits = randn(0, [2, 3])\nprobs = softmax(logits, 1)\n",
        &mut env,
    );
    assert_eq!(env.get_tag("probs"), Some(&ValueTag::Probability));
}

#[test]
fn sigmoid_tags_result_as_probability() {
    let mut env = Environment::new();
    run("x = randn(0, [4])\np = sigmoid(x)\n", &mut env);
    assert_eq!(env.get_tag("p"), Some(&ValueTag::Probability));
}

#[test]
fn cross_entropy_tags_result_as_loss_xent() {
    let mut env = Environment::new();
    // cross_entropy(logits[N, V], targets[N]) -> scalar loss
    env.set("L".to_string(), {
        DenseArray::new(
            mlpl_array::Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 0.5, 0.4, 0.1],
        )
        .unwrap()
    });
    env.set("T".to_string(), DenseArray::from_vec(vec![0.0, 2.0]));
    run("loss = cross_entropy(L, T)\n", &mut env);
    assert_eq!(
        env.get_tag("loss"),
        Some(&ValueTag::Loss {
            kind: LossKind::CrossEntropy
        })
    );
}

#[test]
fn cosine_schedule_tags_result_as_learning_rate() {
    let mut env = Environment::new();
    run("lr = cosine_schedule(0, 100, 0.001, 0.01)\n", &mut env);
    assert_eq!(env.get_tag("lr"), Some(&ValueTag::LearningRate));
}

#[test]
fn linear_warmup_tags_result_as_learning_rate() {
    let mut env = Environment::new();
    run("lr = linear_warmup(5, 10, 0.01)\n", &mut env);
    assert_eq!(env.get_tag("lr"), Some(&ValueTag::LearningRate));
}

#[test]
fn grad_tags_result_as_gradient_with_wrt_name() {
    let mut env = Environment::new();
    let src = "W = param[2, 3]\nX = randn(0, [4, 2])\nY = matmul(X, W)\nloss = mean(Y)\ng = grad(loss, W)\n";
    run(src, &mut env);
    assert_eq!(
        env.get_tag("g"),
        Some(&ValueTag::Gradient {
            wrt: "W".to_string(),
        })
    );
}

#[test]
fn untagged_producer_yields_no_tag() {
    let mut env = Environment::new();
    run(
        "a = randn(0, [2, 3])\nb = randn(1, [3, 2])\nc = matmul(a, b)\n",
        &mut env,
    );
    assert!(env.get_tag("c").is_none());
}

#[test]
fn untagged_inputs_do_not_block_producer_tag() {
    // softmax over an untagged array still produces a Probability binding.
    let mut env = Environment::new();
    run("x = randn(0, [2, 3])\np = softmax(x, 1)\n", &mut env);
    assert_eq!(env.get_tag("p"), Some(&ValueTag::Probability));
    assert!(env.get_tag("x").is_none());
}

#[test]
fn rebinding_overwrites_previous_tag() {
    // Re-assign with a different producer; the new tag wins.
    let mut env = Environment::new();
    let src = "x = randn(0, [2, 3])\nv = sigmoid(x)\nv = softmax(x, 1)\n";
    run(src, &mut env);
    // sigmoid -> Probability, softmax -> Probability; both same tag,
    // but the rebinding should fire and the tag should still be set.
    assert_eq!(env.get_tag("v"), Some(&ValueTag::Probability));
}

#[test]
fn rebinding_to_untagged_producer_keeps_old_tag() {
    // Tags follow the binding NAME, not the value. Re-binding to
    // a non-producer expression keeps the old tag because the
    // step-002 rule never *clears* a tag, only sets one. Step 005
    // adds the user-facing :untag for deliberate clearing.
    let mut env = Environment::new();
    let src = "x = randn(0, [2, 3])\nv = sigmoid(x)\nv = matmul(x, transpose(x))\n";
    run(src, &mut env);
    assert_eq!(env.get_tag("v"), Some(&ValueTag::Probability));
}
