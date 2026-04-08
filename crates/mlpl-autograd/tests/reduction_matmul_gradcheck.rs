//! Finite-difference gradcheck for reductions, shape ops, and matmul.
//!
//! `backward()` seeds the root gradient with ones; for scalar-valued
//! roots that is equivalent to differentiating the scalar itself, and
//! for array-valued roots it differentiates `sum(y)` wrt inputs.

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};

const EPS: f64 = 1e-6;
const TOL: f64 = 1e-4;

fn fd_grad<F>(x: &DenseArray, f: F) -> DenseArray
where
    F: Fn(&DenseArray) -> f64,
{
    let n = x.data().len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut plus = x.data().to_vec();
        let mut minus = x.data().to_vec();
        plus[i] += EPS;
        minus[i] -= EPS;
        let fp = f(&DenseArray::new(x.shape().clone(), plus).unwrap());
        let fm = f(&DenseArray::new(x.shape().clone(), minus).unwrap());
        out[i] = (fp - fm) / (2.0 * EPS);
    }
    DenseArray::new(x.shape().clone(), out).unwrap()
}

fn sum_data(t: &Tensor) -> f64 {
    t.value().data().iter().sum()
}

fn assert_close(a: &DenseArray, b: &DenseArray) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch {a:?} vs {b:?}");
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        assert!((x - y).abs() < TOL, "grad mismatch: {x} vs {y}");
    }
}

#[test]
fn grad_sum_vector() {
    let x_arr = DenseArray::from_vec(vec![1.0, -2.0, 3.0, 0.5]);
    let tape = Tape::new();
    let x = Tensor::param(tape, x_arr.clone());
    let y = x.sum();
    y.backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&t.sum())
    });
    assert_close(&ag, &num);
}

#[test]
fn grad_mean_vector() {
    let x_arr = DenseArray::from_vec(vec![1.0, -2.0, 3.0, 0.5]);
    let tape = Tape::new();
    let x = Tensor::param(tape, x_arr.clone());
    let y = x.mean();
    y.backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&t.mean())
    });
    assert_close(&ag, &num);
}

#[test]
fn grad_softmax_vector() {
    let x_arr = DenseArray::from_vec(vec![0.3, -1.1, 2.0, 0.7]);
    let tape = Tape::new();
    let x = Tensor::param(tape, x_arr.clone());
    // Reduce to scalar via sum so the seed-ones root matches fd.
    let y = x.softmax().sum();
    y.backward();
    let ag = x.grad().expect("grad");
    // softmax outputs always sum to 1, so d/dx sum(softmax(x)) is 0, but
    // gradcheck a non-trivial scalar: sum(log(softmax(x))).
    // Redo the check against a non-trivial loss:
    let tape2 = Tape::new();
    let x2 = Tensor::param(tape2, x_arr.clone());
    let y2 = x2.softmax().log().sum();
    y2.backward();
    let ag2 = x2.grad().expect("grad2");
    let num2 = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&t.softmax().log())
    });
    assert_close(&ag2, &num2);
    // Sanity: sum(softmax) grad should be approximately zero.
    for v in ag.data() {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn grad_softmax_matrix_rows() {
    let x_arr =
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.2, 1.0, -0.5, 1.2, -0.8, 0.3]).unwrap();
    let tape = Tape::new();
    let x = Tensor::param(tape, x_arr.clone());
    let y = x.softmax().log().sum();
    y.backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&t.softmax().log())
    });
    assert_close(&ag, &num);
}

#[test]
fn grad_transpose_matrix() {
    let x_arr =
        DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), x_arr.clone());
    let s = Tensor::leaf(tape, DenseArray::from_scalar(2.0), false);
    let y = x.transpose().mul(&s);
    let loss = y.sum();
    loss.backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape.clone(), xv.clone(), false);
        let s = Tensor::leaf(tape, DenseArray::from_scalar(2.0), false);
        sum_data(&t.transpose().mul(&s))
    });
    assert_close(&ag, &num);
}

#[test]
fn grad_reshape_matrix() {
    let x_arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tape = Tape::new();
    let x = Tensor::param(tape.clone(), x_arr.clone());
    let s = Tensor::leaf(tape, DenseArray::from_scalar(3.0), false);
    let y = x.reshape(Shape::new(vec![2, 3])).mul(&s);
    y.sum().backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape.clone(), xv.clone(), false);
        let s = Tensor::leaf(tape, DenseArray::from_scalar(3.0), false);
        sum_data(&t.reshape(Shape::new(vec![2, 3])).mul(&s))
    });
    assert_close(&ag, &num);
}

#[test]
fn grad_matmul_mm() {
    let a_arr =
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.1, -0.4, 0.7, 0.2, 0.5, -0.3]).unwrap();
    let b_arr =
        DenseArray::new(Shape::new(vec![3, 2]), vec![1.0, -0.5, 0.3, 0.9, -1.1, 0.4]).unwrap();
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), a_arr.clone());
    let b = Tensor::param(tape, b_arr.clone());
    let y = a.matmul(&b).sum();
    y.backward();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    let num_a = fd_grad(&a_arr, |av| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), av.clone(), false);
        let bt = Tensor::leaf(tape, b_arr.clone(), false);
        sum_data(&at.matmul(&bt))
    });
    let num_b = fd_grad(&b_arr, |bv| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), a_arr.clone(), false);
        let bt = Tensor::leaf(tape, bv.clone(), false);
        sum_data(&at.matmul(&bt))
    });
    assert_close(&ga, &num_a);
    assert_close(&gb, &num_b);
}

#[test]
fn grad_matmul_matrix_vector() {
    let a_arr =
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.1, -0.4, 0.7, 0.2, 0.5, -0.3]).unwrap();
    let x_arr = DenseArray::from_vec(vec![0.3, -0.6, 1.2]);
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), a_arr.clone());
    let x = Tensor::param(tape, x_arr.clone());
    let y = a.matmul(&x).sum();
    y.backward();
    let ga = a.grad().unwrap();
    let gx = x.grad().unwrap();
    let num_a = fd_grad(&a_arr, |av| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), av.clone(), false);
        let xt = Tensor::leaf(tape, x_arr.clone(), false);
        sum_data(&at.matmul(&xt))
    });
    let num_x = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), a_arr.clone(), false);
        let xt = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&at.matmul(&xt))
    });
    assert_close(&ga, &num_a);
    assert_close(&gx, &num_x);
}

#[test]
fn grad_two_layer_linear_relu_sum() {
    // x: [1,3], W1: [3,4], W2: [4,2], loss = sum(relu(x @ W1) @ W2)
    let x_arr = DenseArray::new(Shape::new(vec![1, 3]), vec![0.5, -0.2, 1.1]).unwrap();
    let w1_arr = DenseArray::new(
        Shape::new(vec![3, 4]),
        vec![
            0.1, -0.3, 0.2, 0.4, 0.5, 0.0, -0.6, 0.7, -0.2, 0.9, 0.3, -0.4,
        ],
    )
    .unwrap();
    let w2_arr = DenseArray::new(
        Shape::new(vec![4, 2]),
        vec![0.2, -0.5, 0.1, 0.3, -0.4, 0.6, 0.7, -0.1],
    )
    .unwrap();
    let tape = Tape::new();
    let x = Tensor::leaf(tape.clone(), x_arr.clone(), false);
    let w1 = Tensor::param(tape.clone(), w1_arr.clone());
    let w2 = Tensor::param(tape, w2_arr.clone());
    let h = x.matmul(&w1).relu();
    let y = h.matmul(&w2);
    let loss = y.sum();
    loss.backward();
    let gw1 = w1.grad().unwrap();
    let gw2 = w2.grad().unwrap();

    let num_w1 = fd_grad(&w1_arr, |wv| {
        let tape = Tape::new();
        let xt = Tensor::leaf(tape.clone(), x_arr.clone(), false);
        let w1t = Tensor::leaf(tape.clone(), wv.clone(), false);
        let w2t = Tensor::leaf(tape, w2_arr.clone(), false);
        sum_data(&xt.matmul(&w1t).relu().matmul(&w2t))
    });
    let num_w2 = fd_grad(&w2_arr, |wv| {
        let tape = Tape::new();
        let xt = Tensor::leaf(tape.clone(), x_arr.clone(), false);
        let w1t = Tensor::leaf(tape.clone(), w1_arr.clone(), false);
        let w2t = Tensor::leaf(tape, wv.clone(), false);
        sum_data(&xt.matmul(&w1t).relu().matmul(&w2t))
    });
    assert_close(&gw1, &num_w1);
    assert_close(&gw2, &num_w2);
}
