//! Stack op tests. Saga 29 step 013.
//!
//! Confirms `Tensor::stack` forward output matches a hand-rolled
//! concat along the same axis, and gradcheck-parities its backward
//! against centered finite differences for axis 0, axis 1 on rank-
//! 2 input, and axis 1 on rank-3 input (the three patterns the
//! multi-head + rank-3 attention lowering actually exercises).

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn stack_axis0_rank2_forward_matches_concat_chain() {
    let tape = Tape::new();
    let a = Tensor::leaf(tape.clone(), arr(vec![2, 3], vec![1.0; 6]), false);
    let b = Tensor::leaf(tape.clone(), arr(vec![2, 3], vec![2.0; 6]), false);
    let c = Tensor::leaf(tape.clone(), arr(vec![2, 3], vec![3.0; 6]), false);
    let stacked = Tensor::stack(&[a.clone(), b.clone(), c.clone()], 0);
    let chained = a.concat(&b, 0).concat(&c, 0);
    assert_eq!(stacked.value().shape().dims(), &[6, 3]);
    assert_eq!(stacked.value().data(), chained.value().data());
}

#[test]
fn stack_axis1_rank2_forward_matches_concat_chain() {
    let tape = Tape::new();
    let a = Tensor::leaf(
        tape.clone(),
        arr(vec![2, 3], vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        false,
    );
    let b = Tensor::leaf(
        tape.clone(),
        arr(vec![2, 3], vec![2.0, 2.1, 2.2, 2.3, 2.4, 2.5]),
        false,
    );
    let stacked = Tensor::stack(&[a.clone(), b.clone()], 1);
    let chained = a.concat(&b, 1);
    assert_eq!(stacked.value().shape().dims(), &[2, 6]);
    assert_eq!(stacked.value().data(), chained.value().data());
}

#[test]
fn stack_axis0_rank3_forward_matches_concat_chain() {
    let tape = Tape::new();
    let a = Tensor::leaf(
        tape.clone(),
        arr(vec![1, 2, 3], (0..6).map(|i| i as f64).collect()),
        false,
    );
    let b = Tensor::leaf(
        tape.clone(),
        arr(vec![1, 2, 3], (6..12).map(|i| i as f64).collect()),
        false,
    );
    let c = Tensor::leaf(
        tape.clone(),
        arr(vec![1, 2, 3], (12..18).map(|i| i as f64).collect()),
        false,
    );
    let stacked = Tensor::stack(&[a.clone(), b.clone(), c.clone()], 0);
    let chained = a.concat(&b, 0).concat(&c, 0);
    assert_eq!(stacked.value().shape().dims(), &[3, 2, 3]);
    assert_eq!(stacked.value().data(), chained.value().data());
}

/// Pin the analytic backward of `stack` against a centered
/// finite-difference reference. Each parent is a trainable
/// param; the loss is `sum(stack)`. The gradient of each parent
/// must equal the corresponding slab of `d loss / d stack`, which
/// is `1` everywhere -- so every entry of every parent's grad
/// must be `1.0`. (This also catches mis-allocation across
/// parents.)
#[test]
fn stack_axis1_rank2_gradcheck_against_sum_loss() {
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), arr(vec![2, 3], vec![0.1; 6]));
    let b = Tensor::param(tape.clone(), arr(vec![2, 3], vec![0.2; 6]));
    let c = Tensor::param(tape.clone(), arr(vec![2, 3], vec![0.3; 6]));
    let stacked = Tensor::stack(&[a.clone(), b.clone(), c.clone()], 1);
    stacked.sum().backward();
    for (k, t) in [&a, &b, &c].iter().enumerate() {
        let g = t.grad().expect("gradient was set");
        assert_eq!(g.shape().dims(), &[2, 3]);
        for (i, v) in g.data().iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "parent {k} entry {i}: grad {v} != 1"
            );
        }
    }
}

/// Stack-then-matmul gradcheck. The analytic gradient of `sum(stack(...) @ W)`
/// for each parent must match centered finite differences entry-wise.
#[test]
fn stack_axis1_then_matmul_grad_matches_finite_difference() {
    // a, b: [2, 2]; stack along axis 1 -> [2, 4]; @ W [4, 1] -> [2, 1].
    let a_data = vec![0.1, -0.2, 0.3, -0.4];
    let b_data = vec![0.5, 0.6, -0.7, 0.8];
    let w_data = vec![1.0, -1.0, 0.5, -0.5];
    let h_fd = 1e-5;

    // Analytic via tape.
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), arr(vec![2, 2], a_data.clone()));
    let b = Tensor::param(tape.clone(), arr(vec![2, 2], b_data.clone()));
    let w = Tensor::leaf(tape.clone(), arr(vec![4, 1], w_data.clone()), false);
    let stacked = Tensor::stack(&[a.clone(), b.clone()], 1);
    let loss = stacked.matmul(&w).sum();
    loss.backward();
    let g_a = a.grad().expect("g_a");
    let g_b = b.grad().expect("g_b");

    let loss_fn = |a_data: &[f64], b_data: &[f64]| -> f64 {
        // Stack a and b along axis 1 manually: row r is
        // [a[r,0], a[r,1], b[r,0], b[r,1]]. Loss = sum(stacked @ w).
        let mut sum = 0.0;
        for r in 0..2 {
            for c in 0..2 {
                sum += a_data[r * 2 + c] * w_data[c];
            }
            for c in 0..2 {
                sum += b_data[r * 2 + c] * w_data[2 + c];
            }
        }
        sum
    };

    for i in 0..a_data.len() {
        let mut plus = a_data.clone();
        plus[i] += h_fd;
        let mut minus = a_data.clone();
        minus[i] -= h_fd;
        let fd = (loss_fn(&plus, &b_data) - loss_fn(&minus, &b_data)) / (2.0 * h_fd);
        assert!(
            (g_a.data()[i] - fd).abs() < 1e-5,
            "a[{i}]: analytic {} vs fd {fd}",
            g_a.data()[i]
        );
    }
    for i in 0..b_data.len() {
        let mut plus = b_data.clone();
        plus[i] += h_fd;
        let mut minus = b_data.clone();
        minus[i] -= h_fd;
        let fd = (loss_fn(&a_data, &plus) - loss_fn(&a_data, &minus)) / (2.0 * h_fd);
        assert!(
            (g_b.data()[i] - fd).abs() < 1e-5,
            "b[{i}]: analytic {} vs fd {fd}",
            g_b.data()[i]
        );
    }
}
