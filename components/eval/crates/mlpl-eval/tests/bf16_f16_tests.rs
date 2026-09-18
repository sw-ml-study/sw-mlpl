//! RS6 (../reasoning-from-scratch): bf16 and f16 dtypes for reinterpret +
//! read, so raw model-weight bytes decode to f64. Decode-only (packing to
//! these half formats is not supported); the caller reads a file's bytes and
//! reinterprets them.

use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

/// Decode the little-endian half-float `bytes` under `dtype` via
/// reinterpret + `read_<dtype>` at byte 0.
fn decode(bytes: &[u8], dtype: &str) -> f64 {
    let list = bytes
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let src = format!("read_{dtype}(reinterpret(pack([{list}], \"u8\"), \"{dtype}\"), 0)");
    eval(&src).unwrap().data()[0]
}

#[test]
fn reinterpret_accepts_bf16_and_f16() {
    assert!(eval("size_bytes(reinterpret(pack([0, 0], \"u8\"), \"bf16\"))").is_ok());
    assert!(eval("size_bytes(reinterpret(pack([0, 0], \"u8\"), \"f16\"))").is_ok());
    // Width is 2: an odd byte count is rejected.
    assert!(eval("size_bytes(reinterpret(pack([0, 0, 0], \"u8\"), \"bf16\"))").is_err());
}

#[test]
fn bf16_decodes_known_patterns() {
    // bf16 = top 16 bits of the f32 bit pattern. LE bytes.
    assert!((decode(&[0x80, 0x3F], "bf16") - 1.0).abs() < 1e-9); // 0x3F80
    assert!((decode(&[0x00, 0x40], "bf16") - 2.0).abs() < 1e-9); // 0x4000
    assert!((decode(&[0x80, 0xBF], "bf16") + 1.0).abs() < 1e-9); // 0xBF80 = -1.0
    assert!(decode(&[0x00, 0x00], "bf16").abs() < 1e-30); // +0.0
}

#[test]
fn f16_decodes_known_patterns() {
    assert!((decode(&[0x00, 0x3C], "f16") - 1.0).abs() < 1e-9); // 0x3C00
    assert!((decode(&[0x00, 0x40], "f16") - 2.0).abs() < 1e-9); // 0x4000
    assert!((decode(&[0x00, 0xC0], "f16") + 2.0).abs() < 1e-9); // 0xC000 = -2.0
    assert!((decode(&[0x00, 0x35], "f16") - 0.3125).abs() < 1e-6); // 0x3500
}

#[test]
fn f16_handles_inf_nan_and_subnormal() {
    assert!(decode(&[0x00, 0x7C], "f16").is_infinite() && decode(&[0x00, 0x7C], "f16") > 0.0); // 0x7C00
    assert!(decode(&[0x00, 0x7E], "f16").is_nan()); // 0x7E00
    // 0x0001 = smallest positive subnormal = 2^-24 ~= 5.96e-8.
    assert!((decode(&[0x01, 0x00], "f16") - 2f64.powi(-24)).abs() < 1e-12);
}

#[test]
fn packing_to_half_is_rejected() {
    let err = eval("pack([1, 2], \"bf16\")").unwrap_err();
    assert!(
        format!("{err}").contains("bf16"),
        "actionable decode-only error"
    );
}
