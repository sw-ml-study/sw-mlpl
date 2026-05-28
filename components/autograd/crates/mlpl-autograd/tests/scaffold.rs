//! Scaffold tests: leaf construction and node ids.

use mlpl_array::DenseArray;
use mlpl_autograd::{Tape, Tensor};

#[test]
fn construct_param_tensor() {
    let tape = Tape::new();
    let t = Tensor::param(tape.clone(), DenseArray::from_scalar(3.0));
    assert!(t.requires_grad());
    assert_eq!(t.value().data(), &[3.0]);
    assert!(t.grad().is_none());
    assert_eq!(tape.len(), 1);
}

#[test]
fn leaf_non_trainable() {
    let tape = Tape::new();
    let t = Tensor::leaf(tape, DenseArray::from_vec(vec![1.0, 2.0]), false);
    assert!(!t.requires_grad());
    assert_eq!(t.value().data(), &[1.0, 2.0]);
}

#[test]
fn backward_on_leaf_seeds_identity() {
    let tape = Tape::new();
    let t = Tensor::param(tape, DenseArray::from_scalar(1.0));
    t.backward();
    // Backward seeds d x / d x = 1 even on a bare leaf.
    assert_eq!(t.grad().unwrap().data(), &[1.0]);
}

#[test]
fn node_ids_are_unique() {
    let tape = Tape::new();
    let a = Tensor::param(tape.clone(), DenseArray::from_scalar(1.0));
    let b = Tensor::param(tape.clone(), DenseArray::from_scalar(2.0));
    let c = Tensor::leaf(tape.clone(), DenseArray::from_scalar(3.0), false);
    assert_ne!(a.node(), b.node());
    assert_ne!(b.node(), c.node());
    assert_ne!(a.node(), c.node());
    assert_eq!(tape.len(), 3);
}

#[test]
fn fresh_tape_is_empty() {
    let tape = Tape::new();
    assert!(tape.is_empty());
}
