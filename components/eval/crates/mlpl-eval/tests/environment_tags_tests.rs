//! Saga 23 step 001: Environment::tags side table for typed values.

use mlpl_core::{ActivationKind, LossKind, ValueTag};
use mlpl_eval::Environment;
use mlpl_eval::env_api::*;

#[test]
fn new_environment_has_no_tags() {
    let env = Environment::new();
    assert!(env.get_tag("anything").is_none());
    assert_eq!(env.tags_iter().count(), 0);
}

#[test]
fn set_tag_then_get_tag_returns_it() {
    let mut env = Environment::new();
    env.set_tag("logits".to_string(), ValueTag::Logit);
    assert_eq!(env.get_tag("logits"), Some(&ValueTag::Logit));
}

#[test]
fn get_tag_absent_name_returns_none() {
    let env = Environment::new();
    assert!(env.get_tag("does-not-exist").is_none());
}

#[test]
fn set_tag_overwrites_previous_tag_per_name() {
    let mut env = Environment::new();
    env.set_tag("x".to_string(), ValueTag::Logit);
    env.set_tag("x".to_string(), ValueTag::Probability);
    assert_eq!(env.get_tag("x"), Some(&ValueTag::Probability));
}

#[test]
fn clear_tag_removes_tag() {
    let mut env = Environment::new();
    env.set_tag(
        "loss".to_string(),
        ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        },
    );
    env.clear_tag("loss");
    assert!(env.get_tag("loss").is_none());
}

#[test]
fn clear_tag_absent_name_is_noop() {
    let mut env = Environment::new();
    env.clear_tag("never-set");
    assert!(env.get_tag("never-set").is_none());
}

#[test]
fn tags_iter_yields_all_pairs() {
    let mut env = Environment::new();
    env.set_tag("logits".to_string(), ValueTag::Logit);
    env.set_tag(
        "h0".to_string(),
        ValueTag::Activation {
            layer: "encoder.tanh_0".to_string(),
            kind: ActivationKind::Tanh,
        },
    );
    env.set_tag("lr".to_string(), ValueTag::LearningRate);

    let mut names: Vec<String> = env
        .tags_iter()
        .map(|(n, _t): (&String, &ValueTag)| n.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["h0", "logits", "lr"]);
}

#[test]
fn tags_survive_re_set_of_underlying_var() {
    let mut env = Environment::new();
    env.set_tag(
        "W1".to_string(),
        ValueTag::Weight {
            layer: "encoder.linear_1".to_string(),
            name: "W".to_string(),
        },
    );
    let tag_before = env.get_tag("W1").cloned();
    assert!(tag_before.is_some());
}

#[test]
fn tags_independent_of_vars_params_frozen() {
    let env = Environment::new();
    assert_eq!(env.tags_iter().count(), 0);
    assert!(env.get("anything").is_none());
    assert!(!env.is_param("anything"));
    assert!(!env.is_frozen("anything"));
}
