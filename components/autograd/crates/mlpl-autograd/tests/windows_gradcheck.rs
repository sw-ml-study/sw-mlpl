//! `Tensor::windows` forward + backward (CNN Phase 5). windows is an
//! OVERLAPPING sliding-window gather, so its gradient is scatter-ADD:
//! each output gradient accumulates into every input position that
//! window covered.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;
use mlpl_autograd::{Tape, Tensor};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn windows_forward_matches_array_op() {
    let tape = Tape::new();
    let x = Tensor::leaf(
        tape.clone(),
        arr(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        false,
    );
    let y = x.windows(&[3], &[1]);
    assert_eq!(y.value().shape().dims(), &[2, 3]);
    assert_eq!(y.value().data(), &[10.0, 20.0, 30.0, 20.0, 30.0, 40.0]);
}

#[test]
fn windows_backward_accumulates_on_overlap_finite_difference() {
    // loss = sum(windows(x, [3], [1]) * w). Each input element's gradient
    // is the sum of w over every window position that read it -- the
    // overlap is where scatter-ADD (not a plain copy) matters.
    let dims = vec![5];
    let base = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), arr(dims.clone(), base.clone()));
    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let w = Tensor::leaf(tape.clone(), arr(vec![3, 3], w_data.clone()), false);
    let loss = x.windows(&[3], &[1]).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient accumulated");
    let eps = 1e-6;
    for i in 0..base.len() {
        let (mut up, mut dn) = (base.clone(), base.clone());
        up[i] += eps;
        dn[i] -= eps;
        let f = |d: Vec<f64>| -> f64 {
            arr(dims.clone(), d)
                .windows(&[3], &[1])
                .unwrap()
                .data()
                .iter()
                .zip(&w_data)
                .map(|(a, b)| a * b)
                .sum()
        };
        let fd = (f(up) - f(dn)) / (2.0 * eps);
        assert!(
            (g.data()[i] - fd).abs() < 1e-5,
            "elem {i}: {} vs {fd}",
            g.data()[i]
        );
    }
}

#[test]
fn windows_2d_backward_matches_finite_difference() {
    // A 2-D patch stack: windows([3,3], [1,1]) -> [2,2,2,2]. Verifies the
    // scatter-add index math on the multi-axis (rank-changing) case.
    let dims = vec![3, 3];
    let base: Vec<f64> = (0..9).map(|i| i as f64 * 0.5 - 1.0).collect();
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), arr(dims.clone(), base.clone()));
    let w_data: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let w = Tensor::leaf(tape.clone(), arr(vec![2, 2, 2, 2], w_data.clone()), false);
    let loss = x.windows(&[2, 2], &[1, 1]).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient accumulated");
    let eps = 1e-6;
    for i in 0..base.len() {
        let (mut up, mut dn) = (base.clone(), base.clone());
        up[i] += eps;
        dn[i] -= eps;
        let f = |d: Vec<f64>| -> f64 {
            arr(dims.clone(), d)
                .windows(&[2, 2], &[1, 1])
                .unwrap()
                .data()
                .iter()
                .zip(&w_data)
                .map(|(a, b)| a * b)
                .sum()
        };
        let fd = (f(up) - f(dn)) / (2.0 * eps);
        assert!(
            (g.data()[i] - fd).abs() < 1e-5,
            "elem {i}: {} vs {fd}",
            g.data()[i]
        );
    }
}
