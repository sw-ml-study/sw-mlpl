//! Saga 23 step 006: typed `:describe`, `:vars`, `:tags`, `:untag`.

use mlpl_array::{DenseArray, Shape};
use mlpl_core::{ActivationKind, LossKind, ValueTag};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, inspect};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn describe_logit_shows_logit_header() {
    let mut env = Environment::new();
    env.set("L".into(), arr(vec![2, 3], vec![1.0; 6]));
    env.set_tag("L".into(), ValueTag::Logit);
    let out = inspect(&mut env, ":describe L").expect("inspect ok");
    assert!(
        out.contains("L -- Logit"),
        "expected Logit header, got: {out}"
    );
}

#[test]
fn describe_probability_shows_row_sum_invariant() {
    let mut env = Environment::new();
    // Two rows summing to 1.0 each.
    env.set(
        "P".into(),
        arr(vec![2, 3], vec![0.1, 0.7, 0.2, 0.3, 0.3, 0.4]),
    );
    env.set_tag("P".into(), ValueTag::Probability);
    let out = inspect(&mut env, ":describe P").expect("inspect ok");
    assert!(
        out.contains("Probability"),
        "expected Probability header: {out}"
    );
    // Should mention the invariant in some form.
    assert!(
        out.to_lowercase().contains("verified") || out.to_lowercase().contains("invariant"),
        "expected sum invariant note: {out}"
    );
}

#[test]
fn describe_probability_violated_invariant_is_flagged() {
    let mut env = Environment::new();
    env.set(
        "P".into(),
        arr(vec![2, 3], vec![0.1, 0.2, 0.3, 0.5, 0.5, 0.5]), // rows sum to 0.6 and 1.5
    );
    env.set_tag("P".into(), ValueTag::Probability);
    let out = inspect(&mut env, ":describe P").expect("inspect ok");
    assert!(
        out.to_lowercase().contains("violated") || out.to_lowercase().contains("not 1"),
        "expected violated note: {out}"
    );
}

#[test]
fn describe_gradient_shows_wrt() {
    let mut env = Environment::new();
    env.set("g".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set_tag("g".into(), ValueTag::Gradient { wrt: "W1".into() });
    let out = inspect(&mut env, ":describe g").expect("inspect ok");
    assert!(out.contains("Gradient"), "expected Gradient header: {out}");
    assert!(out.contains("W1"), "expected wrt: {out}");
}

#[test]
fn describe_weight_shows_layer_and_name() {
    let mut env = Environment::new();
    env.set("__linear_W_0".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.mark_param("__linear_W_0");
    env.set_tag(
        "__linear_W_0".into(),
        ValueTag::Weight {
            layer: "linear_0".into(),
            name: "W".into(),
        },
    );
    let out = inspect(&mut env, ":describe __linear_W_0").expect("inspect ok");
    assert!(out.contains("Weight"), "expected Weight header: {out}");
    assert!(out.contains("linear_0"), "expected layer: {out}");
}

#[test]
fn describe_bias_shows_layer() {
    let mut env = Environment::new();
    env.set("__linear_b_0".into(), arr(vec![1, 3], vec![0.0; 3]));
    env.set_tag(
        "__linear_b_0".into(),
        ValueTag::Bias {
            layer: "linear_0".into(),
        },
    );
    let out = inspect(&mut env, ":describe __linear_b_0").expect("inspect ok");
    assert!(out.contains("Bias"), "expected Bias header: {out}");
    assert!(out.contains("linear_0"), "expected layer: {out}");
}

#[test]
fn describe_activation_shows_layer_and_kind() {
    let mut env = Environment::new();
    env.set("h".into(), arr(vec![2, 4], vec![0.0; 8]));
    env.set_tag(
        "h".into(),
        ValueTag::Activation {
            layer: "mdl".into(),
            kind: ActivationKind::Tanh,
        },
    );
    let out = inspect(&mut env, ":describe h").expect("inspect ok");
    assert!(out.contains("Activation"), "expected Activation: {out}");
    assert!(
        out.contains("Tanh") || out.contains("tanh"),
        "expected kind: {out}"
    );
}

#[test]
fn describe_loss_shows_kind() {
    let mut env = Environment::new();
    env.set("loss".into(), DenseArray::from_scalar(1.234));
    env.set_tag(
        "loss".into(),
        ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        },
    );
    let out = inspect(&mut env, ":describe loss").expect("inspect ok");
    assert!(out.contains("Loss"), "expected Loss header: {out}");
    assert!(out.contains("CrossEntropy"), "expected kind: {out}");
}

#[test]
fn describe_learning_rate_shows_scalar() {
    let mut env = Environment::new();
    env.set("lr".into(), DenseArray::from_scalar(0.001));
    env.set_tag("lr".into(), ValueTag::LearningRate);
    let out = inspect(&mut env, ":describe lr").expect("inspect ok");
    assert!(out.contains("LearningRate"), "expected LR header: {out}");
}

#[test]
fn describe_labels_shows_num_classes() {
    let mut env = Environment::new();
    env.set("y".into(), DenseArray::from_vec(vec![0.0, 1.0, 2.0]));
    env.set_tag("y".into(), ValueTag::Labels { num_classes: 10 });
    let out = inspect(&mut env, ":describe y").expect("inspect ok");
    assert!(out.contains("Labels"), "expected Labels header: {out}");
    assert!(out.contains("10"), "expected num_classes: {out}");
}

#[test]
fn describe_attention_map_mentions_heatmap() {
    let mut env = Environment::new();
    env.set("att".into(), arr(vec![4, 4], vec![0.0; 16]));
    env.set_tag("att".into(), ValueTag::AttentionMap);
    let out = inspect(&mut env, ":describe att").expect("inspect ok");
    assert!(out.contains("AttentionMap"), "expected AttentionMap: {out}");
    assert!(
        out.to_lowercase().contains("heatmap"),
        "expected heatmap note: {out}"
    );
}

#[test]
fn vars_shows_tag_in_summary() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set_tag("x".into(), ValueTag::Logit);
    let out = inspect(&mut env, ":vars").expect("inspect ok");
    assert!(out.contains("Logit"), "expected tag in :vars: {out}");
}

#[test]
fn vars_untagged_unchanged() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3], vec![0.0; 6]));
    let out = inspect(&mut env, ":vars").expect("inspect ok");
    // No tag column; the existing untagged rendering must still work.
    assert!(out.contains("x"));
}

#[test]
fn tags_command_lists_tagged_bindings() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set("y".into(), arr(vec![3, 2], vec![0.0; 6]));
    env.set("z".into(), arr(vec![4], vec![0.0; 4]));
    env.set_tag("x".into(), ValueTag::Logit);
    env.set_tag("z".into(), ValueTag::Probability);
    let out = inspect(&mut env, ":tags").expect("inspect ok");
    assert!(out.contains("x"), "x missing: {out}");
    assert!(out.contains("z"), "z missing: {out}");
    assert!(
        !out.contains("y\n") && !out.contains("y "),
        "untagged y in :tags: {out}"
    );
}

#[test]
fn tags_command_empty_env_shows_no_tags() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":tags").expect("inspect ok");
    assert!(
        out.to_lowercase().contains("no"),
        "expected empty msg: {out}"
    );
}

#[test]
fn untag_clears_tag() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3], vec![0.0; 6]));
    env.set_tag("x".into(), ValueTag::Logit);
    let out = inspect(&mut env, ":untag x").expect("inspect ok");
    assert!(env.get_tag("x").is_none(), "tag should be cleared");
    assert!(
        out.to_lowercase().contains("untagged"),
        "expected confirm: {out}"
    );
}

#[test]
fn untag_no_tag_is_noop_with_message() {
    let mut env = Environment::new();
    env.set("x".into(), arr(vec![2, 3], vec![0.0; 6]));
    let out = inspect(&mut env, ":untag x").expect("inspect ok");
    assert!(
        out.contains("no tag") || out.contains("had no"),
        "msg: {out}"
    );
}
