//! Saga 23 step 003: auto-tag model-aware producers.
//!
//! Two rule clusters covered here:
//!
//!   1. Param creation by `linear` / `embed` / `attention`
//!      attaches `Weight` / `Bias` tags on the param names that
//!      those constructors register in `Environment::params`.
//!
//!   2. `apply(model, X)` results auto-tag `Logit` / `Probability`
//!      / `Activation` based on the model's structural tail
//!      (outermost chain's last child, with chain/residual
//!      walked recursively).

use mlpl_array::{DenseArray, Shape};
use mlpl_core::{ActivationKind, ValueTag};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

fn weight_at(env: &Environment, prefix: &str) -> Option<ValueTag> {
    // Find the first param whose name starts with `prefix` and
    // return its tag.
    env.tags_iter()
        .find(|(n, _)| n.starts_with(prefix))
        .map(|(_, t)| t.clone())
}

#[test]
fn linear_tags_w_as_weight_and_b_as_bias() {
    let mut env = Environment::new();
    run("m = linear(2, 4, 0)\n", &mut env);

    let w_tag = weight_at(&env, "__linear_W_");
    assert!(matches!(
        w_tag,
        Some(ValueTag::Weight { ref layer, ref name }) if layer == "linear_0" && name == "W"
    ));
    let b_tag = weight_at(&env, "__linear_b_");
    assert!(matches!(
        b_tag,
        Some(ValueTag::Bias { ref layer }) if layer == "linear_0"
    ));
}

#[test]
fn embed_tags_table_as_weight() {
    let mut env = Environment::new();
    run("e = embed(8, 4, 0)\n", &mut env);
    let tag = weight_at(&env, "__embed_E_");
    assert!(matches!(
        tag,
        Some(ValueTag::Weight { ref layer, ref name }) if layer == "embed_0" && name == "table"
    ));
}

#[test]
fn attention_tags_four_projections_as_weights() {
    let mut env = Environment::new();
    run("a = attention(4, 1, 0)\n", &mut env);

    let wq = weight_at(&env, "__attn_Wq_");
    assert!(matches!(
        wq,
        Some(ValueTag::Weight { ref layer, ref name }) if layer == "attention_0" && name == "W_q"
    ));
    let wk = weight_at(&env, "__attn_Wk_");
    assert!(matches!(
        wk,
        Some(ValueTag::Weight { ref name, .. }) if name == "W_k"
    ));
    let wv = weight_at(&env, "__attn_Wv_");
    assert!(matches!(
        wv,
        Some(ValueTag::Weight { ref name, .. }) if name == "W_v"
    ));
    let wo = weight_at(&env, "__attn_Wo_");
    assert!(matches!(
        wo,
        Some(ValueTag::Weight { ref name, .. }) if name == "W_o"
    ));
}

#[test]
fn apply_with_linear_tail_tags_logit() {
    // model = linear(2, 4, 0); apply(model, X) -> Logit
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![3, 2]), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
    );
    run("m = linear(2, 4, 0)\nout = apply(m, X)\n", &mut env);
    assert_eq!(env.get_tag("out"), Some(&ValueTag::Logit));
}

#[test]
fn apply_with_chain_linear_tail_tags_logit() {
    // chain(linear, tanh_layer, linear) -- tail is Linear -> Logit
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    run(
        "m = chain(linear(2, 4, 0), tanh_layer(), linear(4, 2, 1))\nout = apply(m, X)\n",
        &mut env,
    );
    assert_eq!(env.get_tag("out"), Some(&ValueTag::Logit));
}

#[test]
fn apply_with_softmax_tail_tags_probability() {
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    run(
        "m = chain(linear(2, 3, 0), softmax_layer())\nout = apply(m, X)\n",
        &mut env,
    );
    assert_eq!(env.get_tag("out"), Some(&ValueTag::Probability));
}

#[test]
fn apply_with_tanh_tail_tags_activation_tanh() {
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    run(
        "m = chain(linear(2, 4, 0), tanh_layer())\nout = apply(m, X)\n",
        &mut env,
    );
    let tag = env.get_tag("out");
    assert!(matches!(
        tag,
        Some(ValueTag::Activation {
            kind: ActivationKind::Tanh,
            ..
        })
    ));
}

#[test]
fn apply_with_relu_tail_tags_activation_relu() {
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    run(
        "m = chain(linear(2, 4, 0), relu_layer())\nout = apply(m, X)\n",
        &mut env,
    );
    let tag = env.get_tag("out");
    assert!(matches!(
        tag,
        Some(ValueTag::Activation {
            kind: ActivationKind::Relu,
            ..
        })
    ));
}

#[test]
fn apply_through_residual_walks_inner_tail() {
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 4]), vec![0.0; 8]).unwrap(),
    );
    // residual(chain(linear, relu_layer, linear)) -- tail is Linear -> Logit
    run(
        "m = residual(chain(linear(4, 4, 0), relu_layer(), linear(4, 4, 1)))\nout = apply(m, X)\n",
        &mut env,
    );
    assert_eq!(env.get_tag("out"), Some(&ValueTag::Logit));
}

#[test]
fn apply_with_no_clean_tail_is_untagged() {
    // RmsNorm tail; not a recognized producer for any tag.
    let mut env = Environment::new();
    env.set(
        "X".to_string(),
        DenseArray::new(Shape::new(vec![2, 4]), vec![0.0; 8]).unwrap(),
    );
    run(
        "m = chain(linear(4, 4, 0), rms_norm(4))\nout = apply(m, X)\n",
        &mut env,
    );
    assert!(env.get_tag("out").is_none());
}

#[test]
fn embedding_tail_does_not_tag_logit() {
    // An embedding model on its own doesn't produce logits.
    // Apply takes a token-id input [N], output [N, d_model].
    let mut env = Environment::new();
    env.set("ids".to_string(), DenseArray::from_vec(vec![0.0, 1.0, 2.0]));
    run("m = embed(4, 3, 0)\nout = apply(m, ids)\n", &mut env);
    assert!(env.get_tag("out").is_none());
}
