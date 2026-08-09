//! The extension value/error boundary + panic containment.

use mlpl_extension_abi::{ExtError, ExtValue, call_contained};

#[test]
fn scalar_values_construct_and_compare() {
    assert_eq!(ExtValue::I64(42), ExtValue::I64(42));
    assert_ne!(ExtValue::I64(42), ExtValue::I64(7));
    assert_eq!(ExtValue::Str("hi".into()), ExtValue::Str("hi".into()));
    assert_eq!(
        ExtValue::Bytes(vec![1, 2, 3]),
        ExtValue::Bytes(vec![1, 2, 3])
    );
    // all six variants exist
    let _ = [
        ExtValue::Nil,
        ExtValue::Bool(true),
        ExtValue::I64(1),
        ExtValue::F64(1.5),
        ExtValue::Str("x".into()),
        ExtValue::Bytes(vec![0]),
    ];
}

#[test]
fn call_contained_passes_ok_through() {
    let answer: mlpl_extension_abi::ExtFn =
        std::sync::Arc::new(|_: &[ExtValue]| Ok(ExtValue::I64(42)));
    assert_eq!(call_contained(&answer, &[]), Ok(ExtValue::I64(42)));
}

#[test]
fn call_contained_passes_domain_err_through() {
    let boom: mlpl_extension_abi::ExtFn =
        std::sync::Arc::new(|_: &[ExtValue]| Err(ExtError::new("nope")));
    let e = call_contained(&boom, &[]).unwrap_err();
    assert_eq!(e.message, "nope");
    assert!(!e.panicked);
}

#[test]
fn call_contained_catches_a_panic() {
    let kaboom: mlpl_extension_abi::ExtFn =
        std::sync::Arc::new(|_: &[ExtValue]| panic!("provider bug"));
    let e = call_contained(&kaboom, &[]).unwrap_err();
    assert!(e.panicked, "panic should be contained, not unwound");
}
