//! Saga 10 step 005: moons / circles synthetic dataset built-ins.

use mlpl_array::DenseArray;
use mlpl_runtime::call_builtin;

fn scalar(v: f64) -> DenseArray {
    DenseArray::from_scalar(v)
}

fn call_dataset(name: &str, seed: f64, n: usize, noise: f64) -> DenseArray {
    call_builtin(name, vec![scalar(seed), scalar(n as f64), scalar(noise)]).unwrap()
}

fn check_shape_and_balance(arr: &DenseArray, n: usize) {
    assert_eq!(arr.shape().dims(), &[n, 3]);
    let labels: Vec<f64> = (0..n).map(|i| arr.data()[i * 3 + 2]).collect();
    let zeros = labels.iter().filter(|&&l| l == 0.0).count();
    let ones = labels.iter().filter(|&&l| l == 1.0).count();
    assert_eq!(zeros + ones, n, "labels must all be 0 or 1");
    // Allow off-by-one for odd n.
    assert!(
        zeros.abs_diff(ones) <= 1,
        "labels must be ~balanced, got {zeros}/{ones}"
    );
}

#[test]
fn moons_shape_balance_and_determinism() {
    let a = call_dataset("moons", 7.0, 200, 0.05);
    let b = call_dataset("moons", 7.0, 200, 0.05);
    check_shape_and_balance(&a, 200);
    assert_eq!(a.data(), b.data(), "moons must be deterministic per seed");

    // x, y should sit roughly in [-2, 3] for the unit-scale moons.
    for i in 0..200 {
        let x = a.data()[i * 3];
        let y = a.data()[i * 3 + 1];
        assert!(x > -2.0 && x < 3.0, "x out of range: {x}");
        assert!(y > -2.0 && y < 2.0, "y out of range: {y}");
    }

    // Different seed should produce a different sample.
    let c = call_dataset("moons", 8.0, 200, 0.05);
    assert_ne!(a.data(), c.data());
}

#[test]
fn circles_shape_balance_and_determinism() {
    let a = call_dataset("circles", 3.0, 200, 0.05);
    let b = call_dataset("circles", 3.0, 200, 0.05);
    check_shape_and_balance(&a, 200);
    assert_eq!(a.data(), b.data());

    // Inner radius ~0.5, outer ~1.0; with noise=0.05 keep within (0.2, 1.4).
    for i in 0..200 {
        let x = a.data()[i * 3];
        let y = a.data()[i * 3 + 1];
        let r = (x * x + y * y).sqrt();
        assert!(r > 0.2 && r < 1.4, "radius out of range: {r}");
    }
}
