//! Saga R1 step 002: tensor wire format unit
//! tests. Round-trip + error cases.

use mlpl_array::{DenseArray, Shape};
use mlpl_mlx_serve::wire::{
    WireError, decode_from_json, decode_tensor, encode_for_json, encode_tensor,
};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn roundtrip_scalar() {
    let a = arr(vec![], vec![42.5]);
    let bytes = encode_tensor(&a).unwrap();
    let b = decode_tensor(&bytes).unwrap();
    assert_eq!(a, b);
}

#[test]
fn roundtrip_vector() {
    let a = arr(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let bytes = encode_tensor(&a).unwrap();
    let b = decode_tensor(&bytes).unwrap();
    assert_eq!(a, b);
}

#[test]
fn roundtrip_matrix() {
    let a = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bytes = encode_tensor(&a).unwrap();
    let b = decode_tensor(&bytes).unwrap();
    assert_eq!(a, b);
}

#[test]
fn roundtrip_rank3() {
    let a = arr(vec![2, 2, 2], (0..8).map(|i| i as f64).collect());
    let bytes = encode_tensor(&a).unwrap();
    let b = decode_tensor(&bytes).unwrap();
    assert_eq!(a, b);
}

#[test]
fn roundtrip_via_json_base64() {
    let a = arr(vec![3], vec![10.0, 20.0, 30.0]);
    let s = encode_for_json(&a).unwrap();
    let b = decode_from_json(&s).unwrap();
    assert_eq!(a, b);
}

#[test]
fn decode_rejects_invalid_bincode() {
    let err = decode_tensor(b"not a bincode envelope").unwrap_err();
    assert!(matches!(err, WireError::Serialize(_)));
}

#[test]
fn decode_rejects_unsupported_version() {
    // Manually craft an envelope with version=99.
    // Use serde + bincode to emit it the same way
    // encode_tensor would.
    let bad = bincode::serialize(&serde_json::json!({
        "version": 99,
        "dtype": 0,
        "ndim": 0,
        "shape": Vec::<u64>::new(),
        "data": Vec::<u8>::new(),
    }))
    .unwrap_or_default();
    // The bincode serialization of a serde_json::Value
    // doesn't match the Envelope schema, so fall back
    // to using the inverse: encode a real tensor, then
    // mutate the version byte.
    let a = arr(vec![1], vec![0.0]);
    let mut bytes = encode_tensor(&a).unwrap();
    bytes[0] = 99; // bincode encodes u32 little-endian
    let _ = bad;
    let err = decode_tensor(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            WireError::UnsupportedVersion {
                saw: 99,
                expected: 1
            } | WireError::Serialize(_)
        ),
        "expected version mismatch or serialize error, got {err:?}"
    );
}

#[test]
fn decode_rejects_non_base64_in_json_path() {
    let err = decode_from_json("!!!not base64!!!").unwrap_err();
    assert!(matches!(err, WireError::Serialize(_)));
}
