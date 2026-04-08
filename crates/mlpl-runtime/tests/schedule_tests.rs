//! Saga 10 step 004: learning-rate schedule built-ins.

use mlpl_array::DenseArray;
use mlpl_runtime::call_builtin;

fn scalar(v: f64) -> DenseArray {
    DenseArray::from_scalar(v)
}

fn call(name: &str, vs: Vec<f64>) -> f64 {
    let args: Vec<DenseArray> = vs.into_iter().map(scalar).collect();
    let r = call_builtin(name, args).unwrap();
    assert_eq!(r.rank(), 0);
    r.data()[0]
}

#[test]
fn cosine_schedule_at_boundaries_and_midpoint() {
    // step=0 -> lr_max
    let v0 = call("cosine_schedule", vec![0.0, 100.0, 0.01, 0.1]);
    assert!(
        (v0 - 0.1).abs() < 1e-12,
        "step=0 expected lr_max=0.1, got {v0}"
    );

    // step=total -> lr_min
    let vt = call("cosine_schedule", vec![100.0, 100.0, 0.01, 0.1]);
    assert!(
        (vt - 0.01).abs() < 1e-12,
        "step=total expected lr_min=0.01, got {vt}"
    );

    // step=total/2 -> mean(lr_min, lr_max) = 0.055
    let vm = call("cosine_schedule", vec![50.0, 100.0, 0.01, 0.1]);
    assert!(
        (vm - 0.055).abs() < 1e-12,
        "midpoint expected 0.055, got {vm}"
    );

    // step beyond total clamps to lr_min
    let vc = call("cosine_schedule", vec![200.0, 100.0, 0.01, 0.1]);
    assert!((vc - 0.01).abs() < 1e-12);
}

#[test]
fn linear_warmup_at_boundaries_and_midpoint() {
    // step=0 -> 0
    let v0 = call("linear_warmup", vec![0.0, 100.0, 0.1]);
    assert!(v0.abs() < 1e-12);

    // step=warmup -> lr
    let vw = call("linear_warmup", vec![100.0, 100.0, 0.1]);
    assert!((vw - 0.1).abs() < 1e-12);

    // step=warmup/2 -> lr/2
    let vh = call("linear_warmup", vec![50.0, 100.0, 0.1]);
    assert!((vh - 0.05).abs() < 1e-12);

    // step beyond warmup clamps to lr
    let vc = call("linear_warmup", vec![500.0, 100.0, 0.1]);
    assert!((vc - 0.1).abs() < 1e-12);
}
