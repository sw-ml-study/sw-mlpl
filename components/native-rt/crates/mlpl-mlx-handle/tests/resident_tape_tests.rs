//! The autograd tape on a RESIDENT MLX backend (saga E4 step 003):
//! leaves upload once, forward intermediates stay on the device
//! (`is_dev` on every supported node), structural ops fall back
//! per-op without derailing the chain, and backward gradients match
//! the all-CPU tape within the fp32 tolerance.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::Tensor;
use mlpl_autograd_tape::Tape;
use mlpl_mlx_handle::register_mlx_device_ops;

const FP32_TOL: f64 = 1e-5;

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

/// Loss = mean(tanh(a @ b + a)); returns `(loss_value, grad_a)`.
fn run(resident: bool) -> (DenseArray, DenseArray) {
    let tape = Tape::new();
    if resident {
        register_mlx_device_ops();
        tape.resident.set(true);
    }
    let a = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]));
    let b = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.5, 0.6, 0.7, 0.8]));
    let loss = a.matmul(&b).add(&a).tanh().mean();
    if resident {
        // Every node on the supported path stays on the device.
        for (i, node) in tape.nodes().iter().enumerate() {
            assert!(node.value.is_dev(), "node {i} must be resident");
        }
    }
    loss.backward();
    (loss.value(), a.grad().expect("param grad"))
}

#[test]
fn resident_forward_matches_cpu_gradients() {
    let (cpu_loss, cpu_grad) = run(false);
    let (mlx_loss, mlx_grad) = run(true);
    assert!((cpu_loss.data()[0] - mlx_loss.data()[0]).abs() < FP32_TOL);
    for (c, m) in cpu_grad.data().iter().zip(mlx_grad.data()) {
        assert!((c - m).abs() < FP32_TOL, "grad {c} vs {m}");
    }
}

#[test]
fn structural_fallback_rejoins_the_device_path() {
    register_mlx_device_ops();
    let tape = Tape::new();
    tape.resident.set(true);
    let a = Tensor::param(Rc::clone(&tape), arr(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    // rotate has no device kernel: its node falls back to host...
    let r = a.rotate(1, 0);
    assert!(
        !tape.nodes()[r.node().0].value.is_dev(),
        "rotate is host-side"
    );
    // ...but the next elementwise op re-uploads and stays resident.
    let s = r.add(&a).sum();
    assert!(
        tape.nodes()[s.node().0].value.is_dev(),
        "chain rejoins the device"
    );
    s.backward();
    // d(sum(rotate(a) + a))/da = 2 everywhere.
    let g = a.grad().expect("grad");
    for v in g.data() {
        assert!((v - 2.0).abs() < FP32_TOL);
    }
}

#[test]
fn cpu_tape_stays_bit_exact_without_residency() {
    // Residency off (even with a backend registered in this
    // process): the tape must be the ordinary f64 CPU tape.
    register_mlx_device_ops();
    let tape = Tape::new();
    let a = Tensor::param(Rc::clone(&tape), arr(vec![2], vec![0.25, 0.5]));
    let loss = a.exp().sum();
    assert!(!tape.nodes()[loss.node().0].value.is_dev());
    loss.backward();
    let g = a.grad().unwrap();
    assert_eq!(g.data(), &[0.25f64.exp(), 0.5f64.exp()], "bit-exact f64");
}
