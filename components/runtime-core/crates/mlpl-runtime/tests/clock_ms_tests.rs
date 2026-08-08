//! clock_ms(): monotonic elapsed-milliseconds clock.

use mlpl_runtime::call_builtin;

fn ms() -> f64 {
    call_builtin("clock_ms", vec![]).unwrap().data()[0]
}

#[test]
fn returns_a_finite_scalar() {
    let out = call_builtin("clock_ms", vec![]).unwrap();
    assert_eq!(out.shape().dims(), &[] as &[usize]);
    assert!(out.data()[0].is_finite() && out.data()[0] >= 0.0);
}

#[test]
fn is_monotonic_non_decreasing() {
    let a = ms();
    let b = ms();
    assert!(b >= a, "{b} >= {a}");
}

#[test]
fn measures_positive_elapsed_across_work() {
    let t0 = ms();
    let mut acc = 0.0f64;
    for i in 0..2_000_000u64 {
        acc += (i as f64).sqrt();
    }
    let dt = ms() - t0;
    assert!(
        dt > 0.0,
        "busy work should take measurable time; acc={acc}, dt={dt}"
    );
}

#[test]
fn rejects_arguments() {
    assert!(call_builtin("clock_ms", vec![mlpl_array::DenseArray::from_scalar(1.0)]).is_err());
}
