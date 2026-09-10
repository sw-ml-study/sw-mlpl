//! `Tensor::reduce_sum` backward: a sum over axes broadcasts the
//! gradient back over the reduced axes (CNN reduce-backward).

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn reduce_sum_single_axis_backward_broadcasts() {
    // loss = sum(reduce_sum(x, [1]) * w); reduce_sum over axis 1 of [2,3]
    // gives row sums [2], so d loss / d x[i,j] = w[i] (broadcast over j).
    let tape = Tape::new();
    let x = Tensor::param(
        tape.clone(),
        arr(vec![2, 3], (0..6).map(|i| i as f64).collect()),
    );
    let w = Tensor::leaf(tape.clone(), arr(vec![2], vec![10.0, 100.0]), false);
    let loss = x.reduce_sum(&[1]).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient");
    assert_eq!(g.data(), &[10.0, 10.0, 10.0, 100.0, 100.0, 100.0]);
}

#[test]
fn reduce_sum_multi_axis_backward_finite_difference() {
    // reduce_sum over axes [1,2] of [2,2,2] -> [2]; check against fd.
    let dims = vec![2, 2, 2];
    let base: Vec<f64> = (0..8).map(|i| i as f64 * 0.25 - 1.0).collect();
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), arr(dims.clone(), base.clone()));
    let w = Tensor::leaf(tape.clone(), arr(vec![2], vec![3.0, 7.0]), false);
    let loss = x.reduce_sum(&[1, 2]).mul(&w).sum();
    loss.backward();
    let g = x.grad().expect("gradient");
    let eps = 1e-6;
    for i in 0..base.len() {
        let (mut up, mut dn) = (base.clone(), base.clone());
        up[i] += eps;
        dn[i] -= eps;
        let f = |d: Vec<f64>| -> f64 {
            // reduce_sum over axes 1,2 of [2,2,2]: each of the 2 outer
            // blocks sums its 4 elements; then dot with w.
            let blk = 4;
            (0..2)
                .map(|b| d[b * blk..b * blk + blk].iter().sum::<f64>() * w.value().data()[b])
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
