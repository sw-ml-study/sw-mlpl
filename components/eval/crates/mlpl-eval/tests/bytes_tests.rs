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
