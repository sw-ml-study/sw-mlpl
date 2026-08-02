//! Seam-counter pinning for the resident backward pass (saga E4
//! step 008): a chain of matmul -> transpose -> reshape -> softmax
//! -> sum must run its ENTIRE backward on the device -- zero graph
//! forces (downloads) and zero CPU fallbacks. Lives in its own test
//! binary because the seam counters are process-global.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::Tensor;
use mlpl_autograd_tape::Tape;
use mlpl_mlx_handle::register_mlx_device_ops;
use mlpl_tensor_handle::{seam_reset, seam_snapshot};

/// The seam counters are process-global: every test in this file
/// serializes on this lock so zero-download assertions are not
/// polluted by a sibling thread (the `MLX_TEST_LOCK` idiom).
static SEAM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn scalar_broadcast_binary_backward_stays_resident_without_downloads() {
    let _serial = SEAM_TEST_LOCK.lock().unwrap();
    register_mlx_device_ops();
    let tape = Tape::new();
    tape.resident.set(true);
    let a = Tensor::param(
        Rc::clone(&tape),
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.4, -0.7, 1.3, 0.2, 0.9, -0.1]).unwrap(),
    );
    // Scalar divisor/addend: the attention-scale / eps shapes that
    // profiling showed falling back to the CPU backward arms.
    let s = Tensor::param(Rc::clone(&tape), DenseArray::from_scalar(2.0));
    let loss = a.div(&s).add(&s).mul(&a).sum();
    seam_reset();
    loss.backward();
    let (_, downloads, _, fallbacks) = seam_snapshot();
    assert_eq!(
        downloads, 0,
        "scalar-broadcast backward must not force the graph"
    );
    assert_eq!(fallbacks, 0, "no CPU arm may run on this chain");
    // The scalar grad is unbroadcast on-device back to its shape.
    let sg = tape.nodes()[s.node().0].grad.clone().expect("s grad");
    assert!(sg.is_dev(), "scalar grad stays resident");
    assert!(sg.dims().is_empty(), "rank-0 scalar grad");
}

#[test]
fn scalar_broadcast_binary_backward_matches_cpu_gradients() {
    let _serial = SEAM_TEST_LOCK.lock().unwrap();
    let mut grads: Vec<Vec<f64>> = Vec::new();
    for resident in [false, true] {
        let tape = Tape::new();
        if resident {
            register_mlx_device_ops();
            tape.resident.set(true);
        }
        let a = Tensor::param(
            Rc::clone(&tape),
            DenseArray::new(Shape::new(vec![2, 3]), vec![0.4, -0.7, 1.3, 0.2, 0.9, -0.1]).unwrap(),
        );
        let s = Tensor::param(Rc::clone(&tape), DenseArray::from_scalar(2.0));
        let t = Tensor::param(Rc::clone(&tape), DenseArray::from_scalar(-0.5));
        a.div(&s).add(&t).mul(&a).sub(&t).sum().backward();
        for p in [&a, &s, &t] {
            grads.push(p.grad().expect("grad").data().to_vec());
        }
    }
    for (c, m) in grads[..3].concat().iter().zip(grads[3..].concat().iter()) {
        assert!(
            (c - m).abs() < 1e-5,
            "scalar-broadcast grad parity: {c} vs {m}"
        );
    }
}

#[test]
fn concat_forward_and_backward_stay_resident_without_downloads() {
    let _serial = SEAM_TEST_LOCK.lock().unwrap();
    register_mlx_device_ops();
    let tape = Tape::new();
    tape.resident.set(true);
    let mk = |dims: Vec<usize>, data: Vec<f64>| DenseArray::new(Shape::new(dims), data).unwrap();
    let a = Tensor::param(Rc::clone(&tape), mk(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]));
    let b = Tensor::param(
        Rc::clone(&tape),
        mk(vec![2, 3], vec![0.9, 0.1, -0.5, 1.1, 0.3, -0.2]),
    );
    // The engram gate shape: concat along axis 1, then elementwise.
    let loss = a.concat(&b, 1).sigmoid().sum();
    assert!(
        tape.nodes()[loss.node().0].value.is_dev(),
        "concat forward must stay resident"
    );
    seam_reset();
    loss.backward();
    let (_, downloads, _, fallbacks) = seam_snapshot();
    assert_eq!(downloads, 0, "concat backward must not force the graph");
    assert_eq!(fallbacks, 0, "no CPU arm may run on this chain");
    for (t, dims) in [(&a, vec![2, 2]), (&b, vec![2, 3])] {
        let g = tape.nodes()[t.node().0].grad.clone().expect("param grad");
        assert!(g.is_dev(), "split gradient stays resident");
        assert_eq!(g.dims(), dims);
    }
}

#[test]
fn concat_backward_matches_cpu_gradients() {
    let _serial = SEAM_TEST_LOCK.lock().unwrap();
    let mut grads: Vec<Vec<f64>> = Vec::new();
    for resident in [false, true] {
        let tape = Tape::new();
        if resident {
            register_mlx_device_ops();
            tape.resident.set(true);
        }
        let mk =
            |dims: Vec<usize>, data: Vec<f64>| DenseArray::new(Shape::new(dims), data).unwrap();
        let a = Tensor::param(Rc::clone(&tape), mk(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]));
        let b = Tensor::param(
            Rc::clone(&tape),
            mk(vec![2, 3], vec![0.9, 0.1, -0.5, 1.1, 0.3, -0.2]),
        );
        a.concat(&b, 1)
            .sigmoid()
            .mul(&a.concat(&b, 1))
            .sum()
            .backward();
        grads.push(a.grad().expect("grad a").data().to_vec());
        grads.push(b.grad().expect("grad b").data().to_vec());
    }
    for (c, m) in grads[..2].concat().iter().zip(grads[2..].concat().iter()) {
        assert!((c - m).abs() < 1e-5, "concat grad parity: {c} vs {m}");
    }
}

#[test]
fn structural_backward_stays_resident_without_downloads() {
    let _serial = SEAM_TEST_LOCK.lock().unwrap();
    register_mlx_device_ops();
    let tape = Tape::new();
    tape.resident.set(true);
    let mk = |data: Vec<f64>| DenseArray::new(Shape::new(vec![2, 2]), data).unwrap();
    let a = Tensor::param(Rc::clone(&tape), mk(vec![0.4, -0.7, 1.3, 0.2]));
    let b = Tensor::param(Rc::clone(&tape), mk(vec![0.9, 0.1, -0.5, 1.1]));
    let loss = a
        .matmul(&b)
        .transpose()
        .reshape(Shape::new(vec![4]))
        .softmax()
        .sum();
    seam_reset();
    loss.backward();
    let (_, downloads, _, fallbacks) = seam_snapshot();
    assert_eq!(downloads, 0, "backward must not force the lazy graph");
    assert_eq!(fallbacks, 0, "no CPU arm may run on this chain");
    for id in [a.node().0, b.node().0] {
        let grad = tape.nodes()[id].grad.clone().expect("param grad");
        assert!(grad.is_dev(), "param grad stays resident");
    }
}
