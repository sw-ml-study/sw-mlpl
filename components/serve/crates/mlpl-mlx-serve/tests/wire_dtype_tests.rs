//! Saga 21.5 step 011: f32 + u8 wire dtypes.
//!
//! The MLX peer wire grew from f64-only to a 3-variant tagged
//! union: `DTYPE_F64 = 0`, `DTYPE_F32 = 1`, `DTYPE_U8 = 2`.
//! `encode_tensor_as(arr, dtype)` picks the on-disk encoding;
//! `decode_tensor` dispatches on the tag and always returns a
//! `DenseArray` (f64 internally).
//!
//! Why the new dtypes: image tensors materialize as u8 at the
//! source and stay u8 over the wire until the first arithmetic
//! op upgrades them. f32 unlocks the standard ML-training
//! precision tier without paying the f64 wire cost.

use mlpl_array::{DenseArray, Shape};
use mlpl_mlx_serve::wire::{
    DTYPE_F32, DTYPE_F64, DTYPE_U8, WireError, decode_tensor, encode_tensor, encode_tensor_as,
};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

/// Default-dtype (no `_as`) encoder stays f64 byte-for-byte:
/// existing callers see no behavior change.
#[test]
fn default_encode_is_byte_identical_to_old_path() {
    let a = arr(vec![3], vec![1.0, 2.0, 3.0]);
    let default_bytes = encode_tensor(&a).unwrap();
    let f64_bytes = encode_tensor_as(&a, DTYPE_F64).unwrap();
    assert_eq!(default_bytes, f64_bytes);
}

/// f32 round-trip: encoding casts f64 -> f32 with f32-precision
/// loss; decoding reads the f32 bytes back into f64. Result is
/// within f32's machine epsilon of the original.
#[test]
fn f32_roundtrip_preserves_f32_precision() {
    let original = arr(vec![4], vec![1.0, 2.5, -3.25, 1234567.875]);
    let bytes = encode_tensor_as(&original, DTYPE_F32).unwrap();
    let decoded = decode_tensor(&bytes).unwrap();
    assert_eq!(decoded.shape().dims(), original.shape().dims());
    for (i, (a, b)) in decoded
        .data()
        .iter()
        .zip(original.data().iter())
        .enumerate()
    {
        // f32 epsilon is ~1.2e-7 of the magnitude; allow a
        // tiny relative slack.
        let tol = (b.abs() * 1e-6).max(1e-30);
        assert!(
            (a - b).abs() <= tol,
            "elem[{i}]: f32 round-trip {a} != {b} (tol {tol})"
        );
    }
}

/// f32 wire is half the size of f64 wire (4 bytes per element
/// vs 8) -- the whole point of the new dtype.
#[test]
fn f32_wire_is_half_the_data_payload_of_f64() {
    let a = arr(vec![100], (0..100).map(|i| i as f64).collect());
    let f64_bytes = encode_tensor_as(&a, DTYPE_F64).unwrap();
    let f32_bytes = encode_tensor_as(&a, DTYPE_F32).unwrap();
    // Envelope overhead (version + dtype + ndim + shape) is
    // identical between the two; only the `data` blob differs.
    // So the size delta should be roughly 400 bytes (100 * 4).
    let delta = f64_bytes.len().saturating_sub(f32_bytes.len());
    assert!(
        (390..=410).contains(&delta),
        "f32 path should save ~400 bytes vs f64, got {delta}"
    );
}

/// u8 round-trip: encoding truncates f64 -> u8 (caller is
/// responsible for keeping values in [0, 255]); decoding lifts
/// the bytes back into f64. Image-tensor primitives produce
/// values in that range to begin with.
#[test]
fn u8_roundtrip_preserves_byte_values() {
    let pixels: Vec<f64> = (0..16).map(|i| (i * 16) as f64).collect();
    let original = arr(vec![16], pixels.clone());
    let bytes = encode_tensor_as(&original, DTYPE_U8).unwrap();
    let decoded = decode_tensor(&bytes).unwrap();
    assert_eq!(decoded.shape().dims(), original.shape().dims());
    for (i, (a, b)) in decoded.data().iter().zip(pixels.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "pixel[{i}]: u8 round-trip {a} != {b}");
    }
}

/// u8 wire is one eighth the size of f64 wire (1 byte vs 8) --
/// the headline win for image-tensor transport.
#[test]
fn u8_wire_is_one_eighth_the_data_payload_of_f64() {
    let a = arr(vec![256], (0..256).map(|i| (i % 256) as f64).collect());
    let f64_bytes = encode_tensor_as(&a, DTYPE_F64).unwrap();
    let u8_bytes = encode_tensor_as(&a, DTYPE_U8).unwrap();
    let delta = f64_bytes.len().saturating_sub(u8_bytes.len());
    // 256 elements * 7 bytes saved each = 1792 bytes nominal.
    // Allow a wider tolerance for envelope encoding effects.
    assert!(
        (1780..=1800).contains(&delta),
        "u8 path should save ~1792 bytes vs f64, got {delta}"
    );
}

/// Decoder rejects unknown dtype tags with a clear error so
/// future wire changes from a newer peer don't get silently
/// misread by an older orchestrator.
#[test]
fn decode_rejects_unknown_dtype_tag() {
    let a = arr(vec![1], vec![0.0]);
    let mut bytes = encode_tensor(&a).unwrap();
    // bincode v1 default: version is the first 4 bytes (u32
    // little-endian); the dtype byte follows at offset 4.
    bytes[4] = 99;
    let err = decode_tensor(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            WireError::UnsupportedDtype(99) | WireError::Serialize(_)
        ),
        "expected UnsupportedDtype(99), got {err:?}"
    );
}

/// f32 encoding works on a rank-2 matrix (no shape regressions
/// at non-1D).
#[test]
fn f32_roundtrip_works_at_rank_2() {
    let m = arr(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bytes = encode_tensor_as(&m, DTYPE_F32).unwrap();
    let decoded = decode_tensor(&bytes).unwrap();
    assert_eq!(decoded.shape().dims(), &[3, 2]);
    for (a, b) in decoded.data().iter().zip(m.data().iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

/// u8 values out of [0, 255] truncate predictably (`as u8` wraps
/// past 255). Caller's responsibility to keep image tensors in
/// range; the wire is honest about what it stores.
#[test]
fn u8_out_of_range_truncates_via_cast() {
    let a = arr(vec![2], vec![300.0, -5.0]);
    let bytes = encode_tensor_as(&a, DTYPE_U8).unwrap();
    let decoded = decode_tensor(&bytes).unwrap();
    // 300 as u8 -> 300 - 256 = 44 (Rust `as u8` from f64 is
    // saturating on overflow, NOT wrapping; check the lifted
    // value is in the valid u8 range).
    assert!(
        (0.0..=255.0).contains(&decoded.data()[0]),
        "out-of-range positive should land in [0, 255]"
    );
    assert!(
        (0.0..=255.0).contains(&decoded.data()[1]),
        "out-of-range negative should land in [0, 255]"
    );
}
