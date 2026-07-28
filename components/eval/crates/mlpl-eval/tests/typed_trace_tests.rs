//! Saga 23 step 007: typed trace events.
//!
//! When tracing is on, the events recorded for tagged
//! assignments must carry input_types and output_type. The
//! trace JSON must round-trip through serde and -- for
//! untagged programs -- must serialize identically to the
//! pre-step-007 shape (skip_serializing_if Vec::is_empty /
//! Option::is_none).

use mlpl_array::{DenseArray, Shape};
use mlpl_core::{LossKind, ValueTag};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program_traced};
use mlpl_parser::{lex, parse};
use mlpl_trace::Trace;

fn run_traced(src: &str, env: &mut Environment) -> Trace {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    let mut trace = Trace::new(src.into());
    eval_program_traced(&stmts, env, &mut trace).unwrap();
    trace
}

#[test]
fn untagged_assign_serializes_unchanged() {
    let mut env = Environment::new();
    let trace = run_traced(
        "a = randn(0, [2, 3])\nc = matmul(a, transpose(a))\n",
        &mut env,
    );
    let json = trace.to_json();
    // Untagged events MUST NOT carry input_types/output_type in
    // the JSON output -- skip_serializing_if must omit them.
    assert!(
        !json.contains("input_types"),
        "untagged trace should not contain input_types: {json}"
    );
    assert!(
        !json.contains("output_type"),
        "untagged trace should not contain output_type: {json}"
    );
}

#[test]
fn softmax_assignment_has_probability_output_type() {
    let mut env = Environment::new();
    let trace = run_traced("x = randn(0, [2, 3])\np = softmax(x, 1)\n", &mut env);

    // Find the assign event for `p`.
    let assigns: Vec<_> = trace
        .events()
        .iter()
        .filter(|e| e.op == "assign" && e.output_type.is_some())
        .collect();
    assert!(
        !assigns.is_empty(),
        "expected at least one typed assign event; trace: {:?}",
        trace.events()
    );

    let p_event = assigns
        .iter()
        .find(|e| matches!(e.output_type.as_ref(), Some(ValueTag::Probability)))
        .expect("expected a Probability-tagged assign event");
    // input_types is omitted when no input carries a tag (the
    // skip-when-all-none policy keeps untagged-flavored events
    // close to the pre-step-007 JSON shape).
    assert!(
        p_event.input_types.is_empty(),
        "input_types should be empty when no inputs are tagged: {:?}",
        p_event.input_types
    );
}

#[test]
fn cross_entropy_assign_has_loss_output_type() {
    let mut env = Environment::new();
    env.set(
        "L".into(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap(),
    );
    env.set("T".into(), DenseArray::from_vec(vec![0.0, 1.0]));
    env.set_tag("L".into(), ValueTag::Logit);
    let trace = run_traced("loss = cross_entropy(L, T)\n", &mut env);

    let loss_event = trace
        .events()
        .iter()
        .find(|e| {
            matches!(
                e.output_type.as_ref(),
                Some(ValueTag::Loss {
                    kind: LossKind::CrossEntropy
                })
            )
        })
        .expect("expected Loss(CrossEntropy) on the loss assign event");

    // The first input arg is the identifier `L` which carries
    // the Logit tag we set on env.
    assert_eq!(loss_event.input_types[0], Some(ValueTag::Logit));
    // Second arg `T` was untagged.
    assert_eq!(loss_event.input_types[1], None);
}

#[test]
fn typed_trace_json_round_trip_preserves_tags() {
    let mut env = Environment::new();
    let trace = run_traced("x = randn(0, [3])\np = sigmoid(x)\n", &mut env);
    let json = trace.to_json();
    let back: Trace = serde_json::from_str(&json).expect("parse round-trip");
    let p_event = back
        .events()
        .iter()
        .find(|e| matches!(e.output_type.as_ref(), Some(ValueTag::Probability)))
        .expect("Probability tag survives round-trip");
    assert_eq!(p_event.output_type, Some(ValueTag::Probability));
}

#[test]
fn assign_from_ident_tag_propagates_into_trace() {
    let mut env = Environment::new();
    env.set("L".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    env.set_tag("L".into(), ValueTag::Logit);
    let trace = run_traced("y = L\n", &mut env);

    // `y = L` triggers the Ident-alias propagation in
    // tag_propagate; the assign event should carry Logit on
    // both input_types[0] and output_type.
    let y_event = trace
        .events()
        .iter()
        .find(|e| e.op == "assign" && e.output_type.is_some())
        .expect("expected typed assign");
    assert_eq!(y_event.output_type, Some(ValueTag::Logit));
}
