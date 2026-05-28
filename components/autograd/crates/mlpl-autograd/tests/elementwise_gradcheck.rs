//! Finite-difference gradcheck for elementwise ops.
//!
//! `backward()` seeds the root gradient with ones, which is equivalent
//! to differentiating `sum(y)` wrt the inputs. All gradchecks compare
//! analytic grads to central-difference grads of `sum(forward(x))`.

use mlpl_array::DenseArray;
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

fn check_unary(x_data: Vec<f64>, apply: fn(&Tensor) -> Tensor) {
    let x_arr = DenseArray::from_vec(x_data);
    let tape = Tape::new();
    let x = Tensor::param(tape, x_arr.clone());
    let y = apply(&x);
    y.backward();
    let ag = x.grad().expect("grad");
    let num = fd_grad(&x_arr, |xv| {
        let tape = Tape::new();
        let t = Tensor::leaf(tape, xv.clone(), false);
        sum_data(&apply(&t))
    });
    assert_close(&ag, &num);
}

fn check_binary(a_data: Vec<f64>, b_data: Vec<f64>, apply: fn(&Tensor, &Tensor) -> Tensor) {
    let a_arr = DenseArray::from_vec(a_data);
    let b_arr = DenseArray::from_vec(b_data);
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), a_arr.clone());
    let b = Tensor::param(tape, b_arr.clone());
    let y = apply(&a, &b);
    y.backward();
    let ga = a.grad().expect("ga");
    let gb = b.grad().expect("gb");
    let num_a = fd_grad(&a_arr, |av| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), av.clone(), false);
        let bt = Tensor::leaf(tape, b_arr.clone(), false);
        sum_data(&apply(&at, &bt))
    });
    let num_b = fd_grad(&b_arr, |bv| {
        let tape = Tape::new();
        let at = Tensor::leaf(tape.clone(), a_arr.clone(), false);
        let bt = Tensor::leaf(tape, bv.clone(), false);
        sum_data(&apply(&at, &bt))
    });
    assert_close(&ga, &num_a);
    assert_close(&gb, &num_b);
}

#[test]
fn grad_neg() {
    check_unary(vec![1.0, -2.0, 3.5], |x| x.neg());
}

#[test]
fn grad_exp() {
    check_unary(vec![0.1, -0.5, 1.2], |x| x.exp());
}

#[test]
fn grad_log() {
    check_unary(vec![0.5, 1.5, 3.0], |x| x.log());
}

#[test]
fn grad_relu_mixed() {
    check_unary(vec![-1.5, -0.4, 0.6, 2.1], |x| x.relu());
}

#[test]
fn grad_tanh() {
    check_unary(vec![-0.8, 0.2, 1.5], |x| x.tanh());
}

#[test]
fn grad_sigmoid() {
    check_unary(vec![-1.2, 0.0, 0.7], |x| x.sigmoid());
}

#[test]
fn grad_add() {
    check_binary(vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 2.5], |a, b| a.add(b));
}

#[test]
fn grad_sub() {
    check_binary(vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 2.5], |a, b| a.sub(b));
}

#[test]
fn grad_mul() {
    check_binary(vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 2.5], |a, b| a.mul(b));
}

#[test]
fn grad_div() {
    check_binary(vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 2.5], |a, b| a.div(b));
}

#[test]
fn grad_composition_chain() {
    check_unary(vec![0.1, 0.4, -0.3], |x| {
        let e = x.exp();
        let t = e.tanh();
        let m = t.mul(x);
        m.sigmoid()
    });
}

#[test]
fn grad_scalar_broadcast_mul() {
    let tape = Tape::new();
    let s = Tensor::param(tape.clone(), DenseArray::from_scalar(3.0));
    let v = Tensor::param(tape, DenseArray::from_vec(vec![1.0, 2.0, 4.0]));
    let y = s.mul(&v);
    assert_eq!(y.value().data(), &[3.0, 6.0, 12.0]);
    y.backward();
    assert_eq!(s.grad().unwrap().data(), &[7.0]);
    assert_eq!(s.grad().unwrap().shape().rank(), 0);
    assert_eq!(v.grad().unwrap().data(), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_scalar_broadcast_add() {
    let tape = Tape::new();
    let s = Tensor::param(tape.clone(), DenseArray::from_scalar(5.0));
    let v = Tensor::param(tape, DenseArray::from_vec(vec![1.0, 2.0]));
    let y = s.add(&v);
    y.backward();
    assert_eq!(s.grad().unwrap().data(), &[2.0]);
    assert_eq!(v.grad().unwrap().data(), &[1.0, 1.0]);
}
