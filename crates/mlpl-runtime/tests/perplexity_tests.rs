//! Tests for the `perplexity` builtin.
//!
//! Saga 33 step 028: perplexity is the canonical LM evaluation
//! metric, defined as `exp(cross_entropy(logits, targets))`.
//! The builtin returns the same number as the manual
//! `exp(cross_entropy(...))` composition.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

fn vec1(data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::vector(data.len()), data.to_vec()).unwrap()
}

#[test]
fn perplexity_equals_exp_of_cross_entropy() {
    let logits = mat(
        3,
        4,
        &[2.0, 1.0, 0.5, 0.1, 0.1, 3.0, 2.0, 1.0, 0.5, 0.5, 4.0, 0.2],
    );
    let targets = vec1(&[0.0, 1.0, 2.0]);
    let ce = call_builtin("cross_entropy", vec![logits.clone(), targets.clone()]).unwrap();
    let pp = call_builtin("perplexity", vec![logits, targets]).unwrap();
    assert_eq!(ce.shape().dims(), &[] as &[usize]);
    assert_eq!(pp.shape().dims(), &[] as &[usize]);
    let expected = ce.data()[0].exp();
    let actual = pp.data()[0];
    assert!(
        (actual - expected).abs() < 1e-12,
        "perplexity {actual} != exp(cross_entropy={}) = {expected}",
        ce.data()[0],
    );
}

#[test]
fn perplexity_uniform_predictions_equals_vocab_size() {
    // Uniform logits (all zeros) yield uniform softmax, so
    // cross_entropy = ln(V) and perplexity = V regardless of
    // the target index.
    let v = 5;
    let logits = mat(2, v, &vec![0.0; 2 * v]);
    let targets = vec1(&[0.0, 3.0]);
    let pp = call_builtin("perplexity", vec![logits, targets]).unwrap();
    assert!(
        (pp.data()[0] - v as f64).abs() < 1e-12,
        "uniform-logits perplexity should equal V={v}, got {}",
        pp.data()[0]
    );
}

#[test]
fn perplexity_perfect_predictions_approaches_one() {
    // Very large logit at the target index makes the softmax
    // close to a one-hot at the target, so CE -> 0 and
    // perplexity -> 1.
    let logits = mat(1, 3, &[100.0, 0.0, 0.0]);
    let targets = vec1(&[0.0]);
    let pp = call_builtin("perplexity", vec![logits, targets]).unwrap();
    assert!(
        (pp.data()[0] - 1.0).abs() < 1e-9,
        "near-deterministic correct prediction should give perplexity ~= 1, got {}",
        pp.data()[0]
    );
}

#[test]
fn perplexity_propagates_shape_errors() {
    // Mismatched target length vs logits rows -> error.
    let logits = mat(3, 4, &[0.5; 12]);
    let bad_targets = vec1(&[0.0, 1.0]);
    let result = call_builtin("perplexity", vec![logits, bad_targets]);
    assert!(result.is_err(), "expected shape mismatch error");
}
