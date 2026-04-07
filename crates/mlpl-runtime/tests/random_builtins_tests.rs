//! Tests for seeded `random` and `randn` built-ins.

use mlpl_array::DenseArray;
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn shape_vec(dims: &[usize]) -> DenseArray {
    DenseArray::from_vec(dims.iter().map(|&d| d as f64).collect())
}

#[test]
fn random_shape_is_respected() {
    let out = call_builtin("random", vec![scalar(42.0), shape_vec(&[3, 4])]).unwrap();
    assert_eq!(out.shape().dims(), &[3, 4]);
    assert_eq!(out.data().len(), 12);
}

#[test]
fn random_is_deterministic_for_same_seed() {
    let a = call_builtin("random", vec![scalar(7.0), shape_vec(&[100])]).unwrap();
    let b = call_builtin("random", vec![scalar(7.0), shape_vec(&[100])]).unwrap();
    assert_eq!(a.data(), b.data());
}

#[test]
fn random_differs_for_different_seed() {
    let a = call_builtin("random", vec![scalar(1.0), shape_vec(&[50])]).unwrap();
    let b = call_builtin("random", vec![scalar(2.0), shape_vec(&[50])]).unwrap();
    assert_ne!(a.data(), b.data());
}

#[test]
fn random_values_in_unit_interval() {
    let out = call_builtin("random", vec![scalar(123.0), shape_vec(&[1000])]).unwrap();
    for &v in out.data() {
        assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
    }
}

#[test]
fn randn_shape_and_determinism() {
    let a = call_builtin("randn", vec![scalar(11.0), shape_vec(&[2, 3])]).unwrap();
    let b = call_builtin("randn", vec![scalar(11.0), shape_vec(&[2, 3])]).unwrap();
    assert_eq!(a.shape().dims(), &[2, 3]);
    assert_eq!(a.data(), b.data());
}

#[test]
fn randn_approximate_mean_and_variance() {
    let out = call_builtin("randn", vec![scalar(99.0), shape_vec(&[10_000])]).unwrap();
    let data = out.data();
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    assert!(mean.abs() < 0.1, "mean {mean} not near 0");
    assert!((var - 1.0).abs() < 0.15, "var {var} not near 1");
}

#[test]
fn random_rejects_non_scalar_seed() {
    let err = call_builtin(
        "random",
        vec![DenseArray::from_vec(vec![1.0, 2.0]), shape_vec(&[2])],
    );
    assert!(err.is_err());
}
