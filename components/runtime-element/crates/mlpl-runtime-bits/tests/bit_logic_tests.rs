//! band / bor / bxor / bnot / popcount over f64-integer bit patterns.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_bits::try_call;

fn call(name: &str, args: Vec<DenseArray>) -> DenseArray {
    try_call(name, args).unwrap().unwrap()
}
fn s(val: f64) -> DenseArray {
    DenseArray::from_scalar(val)
}
fn v(xs: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![xs.len()]), xs.to_vec()).unwrap()
}

#[test]
fn logic_ops_on_scalars() {
    assert_eq!(call("band", vec![s(12.0), s(10.0)]).data(), &[8.0]);
    assert_eq!(call("bor", vec![s(12.0), s(10.0)]).data(), &[14.0]);
    assert_eq!(call("bxor", vec![s(12.0), s(10.0)]).data(), &[6.0]);
    assert_eq!(call("bnot", vec![s(10.0), s(8.0)]).data(), &[245.0]);
    assert_eq!(call("popcount", vec![s(255.0)]).data(), &[8.0]);
}

#[test]
fn logic_ops_pervade_and_broadcast() {
    assert_eq!(
        call("band", vec![v(&[6.0, 12.0]), v(&[3.0, 10.0])]).data(),
        &[2.0, 8.0]
    );
    assert_eq!(
        call("band", vec![v(&[255.0, 256.0, 257.0]), s(255.0)]).data(),
        &[255.0, 0.0, 1.0]
    );
    assert_eq!(
        call("popcount", vec![v(&[1.0, 3.0, 7.0])]).data(),
        &[1.0, 2.0, 3.0]
    );
}

#[test]
fn hamming_distance_is_popcount_of_bxor() {
    let x = call("bxor", vec![s(11.0), s(13.0)]);
    assert_eq!(call("popcount", vec![x]).data(), &[2.0]);
}

#[test]
fn non_integer_domain_is_a_loud_error() {
    assert!(try_call("band", vec![s(1.5), s(2.0)]).unwrap().is_err());
    assert!(try_call("band", vec![s(-1.0), s(2.0)]).unwrap().is_err());
    assert!(
        try_call("popcount", vec![s(f64::INFINITY)])
            .unwrap()
            .is_err()
    );
    assert!(
        try_call("popcount", vec![s(9_007_199_254_740_992.0)])
            .unwrap()
            .is_err()
    );
    assert!(try_call("bnot", vec![s(1.0), s(0.0)]).unwrap().is_err());
    assert!(try_call("band", vec![s(1.0)]).unwrap().is_err());
}
