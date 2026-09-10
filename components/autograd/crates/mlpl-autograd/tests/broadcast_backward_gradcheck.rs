//! Comprehensive VALUE gradchecks for broadcast backward (the C6 fix).
//!
//! A binary op whose operands broadcast (rank difference or extent-1
//! axes) must return each operand's gradient SUMMED over the axes it was
//! broadcast along -- otherwise a broadcast operand (e.g. a convolution
//! kernel) keeps the larger output shape and its gradient is wrong. A
//! shape-only assertion misses this (the earlier regression), so every
//! case here checks VALUES against central finite differences.

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

/// Central finite-difference check: for each element of `pdata`, compare
/// the taped gradient to `(f(p+eps) - f(p-eps)) / 2eps`.
fn fd_check(pdata: &[f64], taped: &[f64], f: impl Fn(&[f64]) -> f64) {
    let eps = 1e-6;
    for i in 0..pdata.len() {
        let (mut up, mut dn) = (pdata.to_vec(), pdata.to_vec());
        up[i] += eps;
        dn[i] -= eps;
        let fd = (f(&up) - f(&dn)) / (2.0 * eps);
        assert!(
            (taped[i] - fd).abs() < 1e-4,
            "elem {i}: taped {} vs fd {fd}",
            taped[i]
        );
    }
}

#[test]
fn mul_backward_wrt_lower_rank_operand() {
    // k [2] * y [3,2] -> [3,2]; grad wrt k[c] = sum_b y[b,c].
    let (kd, yd) = (vec![2.0, 3.0], vec![1.0, 10.0, 100.0, 1000.0, 5.0, 7.0]);
    let tape = Tape::new();
    let k = Tensor::param(tape.clone(), arr(vec![2], kd.clone()));
    let y = Tensor::leaf(tape.clone(), arr(vec![3, 2], yd.clone()), false);
    k.mul(&y).sum().backward();
    fd_check(&kd, k.grad().unwrap().data(), |kk| {
        (0..3)
            .map(|b| (0..2).map(|c| kk[c] * yd[b * 2 + c]).sum::<f64>())
            .sum()
    });
}

#[test]
fn mul_backward_wrt_output_shaped_operand() {
    // grad wrt the LARGER operand y (same shape as output): grad_y = k
    // broadcast; verifies the non-broadcast side is unaffected by the fix.
    let (kd, yd) = (vec![2.0, 3.0], vec![1.0, 10.0, 100.0, 1000.0, 5.0, 7.0]);
    let tape = Tape::new();
    let k = Tensor::leaf(tape.clone(), arr(vec![2], kd.clone()), false);
    let y = Tensor::param(tape.clone(), arr(vec![3, 2], yd.clone()));
    k.mul(&y).sum().backward();
    fd_check(&yd, y.grad().unwrap().data(), |yy| {
        (0..3)
            .map(|b| (0..2).map(|c| kd[c] * yy[b * 2 + c]).sum::<f64>())
            .sum()
    });
}

#[test]
fn add_backward_wrt_broadcast_bias() {
    // bias b [2] + y [4,2]; grad wrt b[c] = sum_b 1 = 4 (count of rows).
    let (bd, yd) = (
        vec![0.5, -0.5],
        (0..8).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let tape = Tape::new();
    let b = Tensor::param(tape.clone(), arr(vec![2], bd.clone()));
    let y = Tensor::leaf(tape.clone(), arr(vec![4, 2], yd.clone()), false);
    b.add(&y).sum().backward();
    fd_check(&bd, b.grad().unwrap().data(), |bb| {
        (0..4)
            .map(|r| (0..2).map(|c| bb[c] + yd[r * 2 + c]).sum::<f64>())
            .sum()
    });
}

#[test]
fn div_backward_wrt_broadcast_denominator() {
    // y [3,2] / d [2]; grad wrt d (the broadcast denominator).
    let (dd, yd) = (vec![2.0, 4.0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tape = Tape::new();
    let d = Tensor::param(tape.clone(), arr(vec![2], dd.clone()));
    let y = Tensor::leaf(tape.clone(), arr(vec![3, 2], yd.clone()), false);
    y.div(&d).sum().backward();
    fd_check(&dd, d.grad().unwrap().data(), |ddv| {
        (0..3)
            .map(|b| (0..2).map(|c| yd[b * 2 + c] / ddv[c]).sum::<f64>())
            .sum()
    });
}

#[test]
fn bidirectional_broadcast_backward() {
    // a [3,1] * b [1,4] -> [3,4]; grad wrt a[i] = sum_j b[j].
    let (ad, bd) = (vec![10.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0]);
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), arr(vec![3, 1], ad.clone()));
    let b = Tensor::leaf(tape.clone(), arr(vec![1, 4], bd.clone()), false);
    a.mul(&b).sum().backward();
    fd_check(&ad, a.grad().unwrap().data(), |aa| {
        (0..3)
            .map(|i| (0..4).map(|j| aa[i] * bd[j]).sum::<f64>())
            .sum()
    });
}

#[test]
fn conv_kernel_gradient_through_windows_and_reduce() {
    // The real conv path: loss = sum(reduce_sum(k * windows(x,[2,2]),
    // [2,3,4])), grad wrt the kernel k [1,2,2] -- composes the broadcast
    // multiply, the sliding-window gather, and the multi-axis reduce.
    let xd: Vec<f64> = vec![0.3, -1.1, 0.7, 2.0, 0.1, -0.5, 1.3, 0.9, -0.2];
    let kd = vec![0.5, -1.0, 0.25, 2.0];
    let tape = Tape::new();
    let x = Tensor::leaf(tape.clone(), arr(vec![1, 3, 3], xd.clone()), false);
    let k = Tensor::param(tape.clone(), arr(vec![1, 2, 2], kd.clone()));
    k.mul(&x.windows(&[2, 2], &[1, 1]))
        .reduce_sum(&[2, 3, 4])
        .sum()
        .backward();
    // pure-f64 forward: sum over out positions of sum over the 2x2 window
    // of k[u,v] * x[i+u, j+v] (single channel).
    let f = |kk: &[f64]| -> f64 {
        let mut total = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for u in 0..2 {
                    for v in 0..2 {
                        total += kk[u * 2 + v] * xd[(i + u) * 3 + (j + v)];
                    }
                }
            }
        }
        total
    };
    fd_check(&kd, k.grad().unwrap().data(), f);
}
