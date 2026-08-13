//! Typed packed byte buffers (`Value::Bytes`) and the `pack` builtin.
//! Saga typed-packed-bytes step 001.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env)
}

fn scalar(src: &str) -> f64 {
    match eval(src).unwrap() {
        Value::Array(a) => {
            assert_eq!(a.data().len(), 1, "expected a scalar from {src}");
            a.data()[0]
        }
        other => panic!("expected a scalar array, got {other:?}"),
    }
}

#[test]
fn size_bytes_reports_the_packed_footprint() {
    // Packed buffers: exactly the byte length.
    assert_eq!(scalar("size_bytes(pack([1, 2, 3], \"u8\"))"), 3.0);
    assert_eq!(scalar("size_bytes(pack([1], \"u32\"))"), 4.0);
    assert_eq!(scalar("size_bytes(pack([1, 2], \"f64\"))"), 16.0);
    // A numeric array is f64-backed: 8 bytes per element.
    assert_eq!(scalar("size_bytes([1, 2, 3])"), 24.0);
    // A string list: sum of UTF-8 byte lengths ("ab" + "c" = 3).
    assert_eq!(scalar("size_bytes([\"ab\", \"c\"])"), 3.0);
}

#[test]
fn size_bytes_sums_a_record_including_keys() {
    // record {a: [1,2]} -> key "a" (1 byte) + 2 f64 (16 bytes) = 17.
    assert_eq!(scalar("size_bytes({a: [1, 2]})"), 17.0);
}

#[test]
fn size_bytes_rejects_an_opaque_value() {
    // A builtin reference has no defined byte footprint.
    assert!(matches!(
        eval("size_bytes(:add)"),
        Err(EvalError::Unsupported(_))
    ));
}

#[test]
fn reinterpret_reviews_the_same_bytes_under_a_new_dtype() {
    // 4 u8 bytes [1,0,0,0] viewed as one little-endian u32 -- same bytes.
    match eval("reinterpret(pack([1, 0, 0, 0], \"u8\"), \"u32\")").unwrap() {
        Value::Bytes { dtype, data } => {
            assert_eq!(format!("{dtype}"), "u32");
            assert_eq!(data, vec![1, 0, 0, 0]);
        }
        other => panic!("expected Value::Bytes, got {other:?}"),
    }
}

#[test]
fn reinterpret_keeps_the_byte_length_and_reports_new_element_count() {
    // 8 u8 bytes as f64 -> one element; size unchanged.
    let v = eval("reinterpret(pack([0, 0, 0, 0, 0, 0, 0, 0], \"u8\"), \"f64\")").unwrap();
    assert_eq!(format!("{v}"), "<bytes: f64[1]>");
    assert_eq!(
        scalar("size_bytes(reinterpret(pack([1, 2, 3, 4], \"u8\"), \"u16\"))"),
        4.0
    );
}

#[test]
fn reinterpret_rejects_an_indivisible_byte_length() {
    // 3 bytes are not a whole number of u32 (width 4) values.
    assert!(matches!(
        eval("reinterpret(pack([1, 2, 3], \"u8\"), \"u32\")"),
        Err(EvalError::Unsupported(_))
    ));
}

#[test]
fn reinterpret_rejects_a_non_buffer_first_argument() {
    assert!(matches!(
        eval("reinterpret([1, 2, 3], \"u8\")"),
        Err(EvalError::Unsupported(_))
    ));
}

#[test]
fn pack_u8_packs_one_byte_per_element() {
    match eval("pack([1, 2, 255], \"u8\")").unwrap() {
        Value::Bytes { dtype, data } => {
            assert_eq!(format!("{dtype}"), "u8");
            assert_eq!(data, vec![1, 2, 255]);
        }
        other => panic!("expected Value::Bytes, got {other:?}"),
    }
}

#[test]
fn pack_u32_is_little_endian() {
    // 1 -> 0x00000001 little-endian = [1, 0, 0, 0]
    match eval("pack([1], \"u32\")").unwrap() {
        Value::Bytes { data, .. } => assert_eq!(data, vec![1, 0, 0, 0]),
        other => panic!("expected Value::Bytes, got {other:?}"),
    }
}

#[test]
fn pack_f32_writes_ieee_bits_little_endian() {
    // 1.0f32 = 0x3F800000 little-endian = [0, 0, 0x80, 0x3F]
    match eval("pack([1], \"f32\")").unwrap() {
        Value::Bytes { data, .. } => assert_eq!(data, vec![0, 0, 0x80, 0x3F]),
        other => panic!("expected Value::Bytes, got {other:?}"),
    }
}

#[test]
fn pack_display_shows_dtype_and_element_count() {
    let v = eval("pack([1, 2, 3, 4], \"u16\")").unwrap();
    // element count (4), not the byte length (8).
    assert_eq!(format!("{v}"), "<bytes: u16[4]>");
}

#[test]
fn pack_rejects_a_fractional_value_for_an_int_dtype() {
    assert!(matches!(
        eval("pack([1.5], \"u8\")"),
        Err(EvalError::Unsupported(_))
    ));
}

#[test]
fn pack_rejects_an_out_of_range_value() {
    assert!(matches!(
        eval("pack([256], \"u8\")"),
        Err(EvalError::Unsupported(_))
    ));
    assert!(matches!(
        eval("pack([0 - 1], \"u8\")"),
        Err(EvalError::Unsupported(_))
    ));
}

#[test]
fn a_bytes_value_round_trips_through_a_variable() {
    // Exercises the env storage lane: assign a Bytes value to a name,
    // then read it back by identifier.
    match eval("b = pack([10, 20], \"u8\")\nb").unwrap() {
        Value::Bytes { dtype, data } => {
            assert_eq!(format!("{dtype}"), "u8");
            assert_eq!(data, vec![10, 20]);
        }
        other => panic!("expected Value::Bytes, got {other:?}"),
    }
}

#[test]
fn pack_rejects_an_unknown_dtype() {
    assert!(matches!(
        eval("pack([1], \"u128\")"),
        Err(EvalError::Unsupported(_))
    ));
}
