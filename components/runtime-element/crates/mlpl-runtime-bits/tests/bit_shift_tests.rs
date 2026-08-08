//! shl / shr / bmask / bits / from_bits.

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
fn width_aware_shift_and_mask() {
    // shl within 8 bits wraps: (1 << 9) & 0xFF = 0
    assert_eq!(call("shl", vec![s(1.0), s(9.0), s(8.0)]).data(), &[0.0]);
    // (3 << 2) & 0xFF = 12
    assert_eq!(call("shl", vec![s(3.0), s(2.0), s(8.0)]).data(), &[12.0]);
    // logical right shift
    assert_eq!(call("shr", vec![s(12.0), s(2.0)]).data(), &[3.0]);
    // bmask keeps low bits
    assert_eq!(call("bmask", vec![s(257.0), s(8.0)]).data(), &[1.0]);
    // pack two nibbles into a byte: bor(shl(hi,4,8), lo)
    let hi = call("shl", vec![s(10.0), s(4.0), s(8.0)]);
    let byte = call("bor", vec![hi, s(5.0)]);
    assert_eq!(byte.data(), &[165.0]); // 0xA5
}

#[test]
fn bits_and_from_bits_round_trip_lsb_first() {
    // 13 = 0b1101 -> LSB-first [1,0,1,1]
    let b = call("bits", vec![s(13.0), s(4.0)]);
    assert_eq!(b.data(), &[1.0, 0.0, 1.0, 1.0]);
    // inverse
    assert_eq!(call("from_bits", vec![b]).data(), &[13.0]);
    // wider view zero-extends
    assert_eq!(call("bits", vec![s(1.0), s(3.0)]).data(), &[1.0, 0.0, 0.0]);
}

#[test]
fn shift_and_view_pervade_elementwise() {
    assert_eq!(
        call("shr", vec![v(&[8.0, 16.0, 32.0]), s(3.0)]).data(),
        &[1.0, 2.0, 4.0]
    );
    assert_eq!(
        call("bmask", vec![v(&[255.0, 256.0]), s(8.0)]).data(),
        &[255.0, 0.0]
    );
}

#[test]
fn errors_are_loud() {
    assert!(try_call("shl", vec![s(1.0), s(2.0)]).unwrap().is_err()); // arity
    assert!(
        try_call("shl", vec![s(1.0), s(60.0), s(8.0)])
            .unwrap()
            .is_err()
    ); // count > 53
    assert!(
        try_call("bits", vec![v(&[1.0, 2.0]), s(4.0)])
            .unwrap()
            .is_err()
    ); // non-scalar x
    assert!(try_call("from_bits", vec![s(1.0)]).unwrap().is_err()); // non-vector
    assert!(
        try_call("from_bits", vec![v(&[1.0, 2.0])])
            .unwrap()
            .is_err()
    ); // entry not 0/1
}
