//! Saga 23 step 005: tag propagation through arithmetic,
//! transpose, reshape, reductions, and unary negation.

use mlpl_array::{DenseArray, Shape};
use mlpl_core::{LossKind, ValueTag};
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn try_run(src: &str, env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
}

fn run(src: &str, env: &mut Environment) {
    try_run(src, env).unwrap();
}

fn type_mismatch(err: &EvalError) -> Option<(String, String, String)> {
    match err {
        EvalError::TypeMismatch {
            op,
            expected,
            actual,
            ..
        } => Some((op.clone(), expected.clone(), actual.clone())),
        _ => None,
    }
}

fn arr2(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn logit_plus_logit_yields_logit() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set("B".into(), arr2(vec![2, 3], vec![2.0; 6]));
    env.set_tag("A".into(), ValueTag::Logit);
    env.set_tag("B".into(), ValueTag::Logit);
    run("c = A + B\n", &mut env);
    assert_eq!(env.get_tag("c"), Some(&ValueTag::Logit));
}

#[test]
fn loss_plus_loss_yields_loss() {
    let mut env = Environment::new();
    env.set("L1".into(), DenseArray::from_scalar(1.0));
    env.set("L2".into(), DenseArray::from_scalar(2.0));
    env.set_tag(
        "L1".into(),
        ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        },
    );
    env.set_tag(
        "L2".into(),
        ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        },
    );
    run("total = L1 + L2\n", &mut env);
    assert_eq!(
        env.get_tag("total"),
        Some(&ValueTag::Loss {
            kind: LossKind::CrossEntropy
        })
    );
}

#[test]
fn tagged_plus_untagged_keeps_the_tagged_side() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set("B".into(), arr2(vec![2, 3], vec![2.0; 6]));
    env.set_tag("A".into(), ValueTag::Logit);
    run("c = A + B\n", &mut env);
    assert_eq!(env.get_tag("c"), Some(&ValueTag::Logit));
}

#[test]
fn untagged_plus_tagged_keeps_the_tagged_side() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set("B".into(), arr2(vec![2, 3], vec![2.0; 6]));
    env.set_tag("B".into(), ValueTag::Logit);
    run("c = A + B\n", &mut env);
    assert_eq!(env.get_tag("c"), Some(&ValueTag::Logit));
}

#[test]
fn untagged_plus_untagged_stays_untagged() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set("B".into(), arr2(vec![2, 3], vec![2.0; 6]));
    run("c = A + B\n", &mut env);
    assert!(env.get_tag("c").is_none());
}

#[test]
fn logit_plus_probability_raises_type_mismatch() {
    let mut env = Environment::new();
    env.set("L".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set("P".into(), arr2(vec![2, 3], vec![0.3; 6]));
    env.set_tag("L".into(), ValueTag::Logit);
    env.set_tag("P".into(), ValueTag::Probability);
    let err = try_run("c = L + P\n", &mut env).unwrap_err();
    let (_op, expected, actual) =
        type_mismatch(&err).expect("expected TypeMismatch on Logit + Probability");
    assert!(
        actual.contains("Logit") && actual.contains("Probability"),
        "actual mention both tags: {actual}"
    );
    assert!(expected.contains("compatible"), "expected: {expected}");
}

#[test]
fn loss_plus_probability_raises_type_mismatch() {
    let mut env = Environment::new();
    env.set("L".into(), DenseArray::from_scalar(1.0));
    env.set("P".into(), arr2(vec![2, 3], vec![0.5; 6]));
    env.set_tag(
        "L".into(),
        ValueTag::Loss {
            kind: LossKind::Mse,
        },
    );
    env.set_tag("P".into(), ValueTag::Probability);
    let err = try_run("c = L + P\n", &mut env).unwrap_err();
    assert!(type_mismatch(&err).is_some(), "got: {err:?}");
}

#[test]
fn negation_preserves_tag() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set_tag("A".into(), ValueTag::Logit);
    run("c = -A\n", &mut env);
    assert_eq!(env.get_tag("c"), Some(&ValueTag::Logit));
}

#[test]
fn transpose_preserves_tag() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set_tag("A".into(), ValueTag::Logit);
    run("c = transpose(A)\n", &mut env);
    assert_eq!(env.get_tag("c"), Some(&ValueTag::Logit));
}

#[test]
fn reshape_clears_tag() {
    let mut env = Environment::new();
    env.set("A".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set_tag("A".into(), ValueTag::Logit);
    run("c = reshape(A, [3, 2])\n", &mut env);
    assert!(env.get_tag("c").is_none());
}

#[test]
fn mean_of_loss_survives_as_loss() {
    let mut env = Environment::new();
    env.set("L".into(), arr2(vec![3], vec![1.0, 2.0, 3.0]));
    env.set_tag(
        "L".into(),
        ValueTag::Loss {
            kind: LossKind::Mse,
        },
    );
    run("c = mean(L)\n", &mut env);
    assert_eq!(
        env.get_tag("c"),
        Some(&ValueTag::Loss {
            kind: LossKind::Mse
        })
    );
}

#[test]
fn mean_of_probability_clears_tag() {
    // A partial sum / mean of probabilities is not itself a Probability.
    let mut env = Environment::new();
    env.set("P".into(), arr2(vec![3], vec![0.2, 0.3, 0.5]));
    env.set_tag("P".into(), ValueTag::Probability);
    run("c = mean(P)\n", &mut env);
    assert!(env.get_tag("c").is_none());
}

#[test]
fn reduce_add_of_loss_survives() {
    let mut env = Environment::new();
    env.set("L".into(), arr2(vec![3], vec![1.0, 2.0, 3.0]));
    env.set_tag(
        "L".into(),
        ValueTag::Loss {
            kind: LossKind::CrossEntropy,
        },
    );
    run("c = reduce_add(L, 0)\n", &mut env);
    assert!(matches!(
        env.get_tag("c"),
        Some(ValueTag::Loss {
            kind: LossKind::CrossEntropy
        })
    ));
}

#[test]
fn producer_takes_precedence_over_propagation() {
    // softmax is a Probability producer; even if the input is tagged
    // Logit, the result is the producer's tag, not a propagated one.
    let mut env = Environment::new();
    env.set("L".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set_tag("L".into(), ValueTag::Logit);
    run("p = softmax(L, 1)\n", &mut env);
    assert_eq!(env.get_tag("p"), Some(&ValueTag::Probability));
}

#[test]
fn assign_from_ident_propagates_tag() {
    // y = x where x is tagged should carry the tag through.
    let mut env = Environment::new();
    env.set("L".into(), arr2(vec![2, 3], vec![1.0; 6]));
    env.set_tag("L".into(), ValueTag::Logit);
    run("y = L\n", &mut env);
    assert_eq!(env.get_tag("y"), Some(&ValueTag::Logit));
}
