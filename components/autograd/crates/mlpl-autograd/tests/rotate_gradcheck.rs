//! `Tensor::rotate` forward + backward (Game of Life saga step 1).
//! Rotate is a pure permutation, so its gradient is the inverse
//! rotation of the upstream gradient.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;
use mlpl_autograd::{Tape, Tensor};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn rotate_forward_matches_array_op() {
    let tape = Tape::new();
    let x = Tensor::leaf(tape.clone(), arr(vec![3], vec![1.0, 2.0, 3.0]), false);
    let y = x.rotate(1, 0);
    assert_eq!(y.value().data(), &[2.0, 3.0, 1.0]);
}

#[test]
fn rotate_backward_is_inverse_rotation() {
    // loss = sum(rotate(x, 1, 0) * w)  =>  dloss/dx = rotate(w, -1, 0)
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), arr(vec![3], vec![1.0, 2.0, 3.0]));
    let w = Tensor::leaf(tape.clone(), arr(vec![3], vec![10.0, 20.0, 30.0]), false);
    let loss = x.rotate(1, 0).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient accumulated");
    let expected = arr(vec![3], vec![10.0, 20.0, 30.0]).rotate(-1, 0).unwrap();
    assert_eq!(g.data(), expected.data());
}

#[test]
fn rotate_backward_matrix_axis1_finite_difference() {
    let dims = vec![2, 3];
    let base = vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5];
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), arr(dims.clone(), base.clone()));
    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = Tensor::leaf(tape.clone(), arr(dims.clone(), w_data.clone()), false);
    let loss = x.rotate(2, 1).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient accumulated");
    let eps = 1e-6;
    for i in 0..base.len() {
        let mut up = base.clone();
        let mut dn = base.clone();
        up[i] += eps;
        dn[i] -= eps;
        let f = |d: Vec<f64>| -> f64 {
            arr(dims.clone(), d)
                .rotate(2, 1)
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
