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

/// Per-op backward parity: run `build` on a CPU tape and a resident
/// tape, compare BOTH param grads at fp32 tolerance.
fn backward_parity(build: impl Fn(&Tensor, &Tensor) -> Tensor) {
    let mut grads: Vec<Vec<f64>> = Vec::new();
    for resident in [false, true] {
        let tape = Tape::new();
        if resident {
            register_mlx_device_ops();
            tape.resident.set(true);
        }
        let a = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]));
        let b = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.9, 0.1, -0.5, 1.1]));
        build(&a, &b).backward();
        grads.push(a.grad().expect("grad a").data().to_vec());
        grads.push(b.grad().expect("grad b").data().to_vec());
    }
    for (c, m) in grads[0]
        .iter()
        .chain(&grads[1])
        .zip(grads[2].iter().chain(&grads[3]))
    {
        assert!((c - m).abs() < FP32_TOL, "backward parity: {c} vs {m}");
    }
}

#[test]
fn resident_backward_matches_cpu_for_every_core_op() {
    // Elementwise binaries + matmul.
    backward_parity(|a, b| a.add(b).sum());
    backward_parity(|a, b| a.sub(b).mean());
    backward_parity(|a, b| a.mul(b).sum());
    backward_parity(|a, b| a.div(&b.mul(b).add(b)).sum()); // keep b well away from 0
    backward_parity(|a, b| a.matmul(b).sum());
    // Unaries (chained off both params so both get grads).
    backward_parity(|a, b| a.exp().mul(b).sum());
    backward_parity(|a, b| a.tanh().mul(b).mean());
    backward_parity(|a, b| a.sigmoid().mul(b).sum());
    backward_parity(|a, b| a.relu().mul(b).sum());
    backward_parity(|a, b| a.neg().mul(b).sum());
    backward_parity(|a, b| a.mul(a).add(b).log().sum()); // log of positive-ish
    // Softmax + transpose + reshape in one chain.
    backward_parity(|a, b| a.matmul(b).softmax().transpose().sum());
    backward_parity(|a, b| {
        a.matmul(b)
            .reshape(mlpl_array::Shape::new(vec![4]))
            .softmax()
            .sum()
    });
}

#[test]
fn resident_backward_keeps_grads_on_device() {
    register_mlx_device_ops();
    let tape = Tape::new();
    tape.resident.set(true);
    let a = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]));
    let b = Tensor::param(Rc::clone(&tape), arr(vec![2, 2], vec![0.9, 0.1, -0.5, 1.1]));
    a.matmul(&b).tanh().mean().backward();
    for id in [a.node().0, b.node().0] {
        let node = &tape.nodes()[id];
        assert!(
            node.grad.as_ref().expect("grad present").is_dev(),
            "param grad stays resident on the device"
        );
    }
}

#[test]
fn tiny_lm_shaped_chain_backward_parity() {
    // matmul -> tanh -> matmul -> cross_entropy: CE backward is the
    // documented CPU fallback; the mixed accumulate must still
    // deliver grads that match the all-CPU tape.
    let mut all: Vec<Vec<f64>> = Vec::new();
    for resident in [false, true] {
        let tape = Tape::new();
        if resident {
            register_mlx_device_ops();
            tape.resident.set(true);
        }
        let x = Tensor::leaf(
            Rc::clone(&tape),
            arr(
                vec![3, 4],
                (0..12).map(|i| f64::from(i) * 0.1 - 0.5).collect(),
            ),
            false,
        );
        let w1 = Tensor::param(
            Rc::clone(&tape),
            arr(
                vec![4, 4],
                (0..16).map(|i| f64::from(i) * 0.05 - 0.4).collect(),
            ),
        );
        let w2 = Tensor::param(
            Rc::clone(&tape),
            arr(
                vec![4, 5],
                (0..20).map(|i| 0.3 - f64::from(i) * 0.03).collect(),
            ),
        );
        let loss = x
            .matmul(&w1)
            .tanh()
            .matmul(&w2)
            .cross_entropy(vec![1, 4, 0]);
        loss.backward();
        all.push(w1.grad().unwrap().data().to_vec());
        all.push(w2.grad().unwrap().data().to_vec());
    }
    for (c, m) in all[0]
        .iter()
        .chain(&all[1])
        .zip(all[2].iter().chain(&all[3]))
    {
        assert!((c - m).abs() < FP32_TOL, "tiny-lm chain: {c} vs {m}");
    }
}
