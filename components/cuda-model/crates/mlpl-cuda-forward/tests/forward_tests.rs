//! Parity tests for the traceable forward primitives: each op vs a
//! hand-computed reference within an fp32 tolerance, on the GPU.
//! Triple-gated; a no-op off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use candle_core::{Device, Tensor};
use mlpl_cuda_forward::{
    causal_attention, causal_mask, cross_entropy, embed, lora_linear, rms_norm,
};

const TOL: f64 = 1e-4;

fn dev() -> Device {
    Device::new_cuda(0).expect("CUDA device 0")
}

fn t(data: &[f32], shape: (usize, usize), d: &Device) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, d).unwrap()
}

fn flat(t: &Tensor) -> Vec<f64> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&x| f64::from(x))
        .collect()
}

fn assert_within(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= TOL, "elem {i}: got={g} want={w}");
    }
}

#[test]
fn embed_is_onehot_matmul() {
    let d = dev();
    let onehot = t(&[0., 1., 0., 0., 0., 1.], (2, 3), &d); // rows pick idx 1, 2
    let table = t(&[1., 2., 3., 4., 5., 6.], (3, 2), &d); // [3,2]
    assert_within(
        &flat(&embed(&onehot, &table).unwrap()),
        &[3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn rms_norm_matches_reference() {
    let d = dev();
    let x = t(&[1., 2., 3., 4.], (1, 4), &d);
    let gamma = Tensor::from_vec(vec![1.0f32; 4], 4, &d).unwrap();
    // mean(x^2) = 7.5; x / sqrt(7.5)
    let s = 7.5f64.sqrt();
    let want: Vec<f64> = [1., 2., 3., 4.].iter().map(|v| v / s).collect();
    assert_within(&flat(&rms_norm(&x, &gamma, 1e-8).unwrap()), &want);
}

#[test]
fn cross_entropy_matches_reference() {
    let d = dev();
    let logits = t(&[1., 2., 0., 0., 1., 3.], (2, 3), &d);
    let targets = t(&[0., 1., 0., 0., 0., 1.], (2, 3), &d);
    // row0: log(e+e^2+1) - 2 = 0.40760; row1: log(1+e+e^3) - 3 = 0.16989
    let want = f64::midpoint(0.407_60, 0.169_89);
    let got = flat(&cross_entropy(&logits, &targets).unwrap())[0];
    assert!((got - want).abs() <= TOL, "ce got={got} want={want}");
}

#[test]
fn causal_mask_is_lower_triangular() {
    let m = flat(&causal_mask(2, &dev()).unwrap());
    assert!(
        m[0] == 0.0 && m[2] == 0.0 && m[3] == 0.0,
        "diag/below are 0"
    );
    assert!(m[1] < -1e8, "above-diagonal is large-negative");
}

#[test]
fn causal_attention_identity_weights() {
    let d = dev();
    let x = t(&[1., 0., 0., 1.], (2, 2), &d); // identity rows
    let id = || t(&[1., 0., 0., 1.], (2, 2), &d);
    let mask = causal_mask(2, &d).unwrap();
    let out = causal_attention(&x, &id(), &id(), &id(), &id(), &mask).unwrap();
    // row0 attends only key0 -> [1,0]; row1 softmax([0, 1/sqrt2]) over v=I.
    let w1 = (1.0f64 / 2.0f64.sqrt()).exp();
    let (a0, a1) = (1.0 / (1.0 + w1), w1 / (1.0 + w1));
    assert_within(&flat(&out), &[1.0, 0.0, a0, a1]);
}

#[test]
fn lora_linear_matches_reference() {
    let device = dev();
    let x = t(&[1., 2.], (1, 2), &device);
    let w = t(&[1., 0., 0., 1.], (2, 2), &device);
    let a = t(&[1., 1.], (2, 1), &device);
    let b = t(&[1., 1.], (1, 2), &device);
    // w + 2*(a@b) = [[3,2],[2,3]]; x @ that = [7, 8]
    assert_within(
        &flat(&lora_linear(&x, &w, &a, &b, 2.0).unwrap()),
        &[7.0, 8.0],
    );
}
