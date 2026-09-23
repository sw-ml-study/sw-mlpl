//! `unpack(bytes, dtype)` -- the bulk inverse of `pack`: a typed byte buffer
//! decoded in one native pass to a flat 1-D array (reasoning-from-scratch R11:
//! model weights stored as bf16 must load without one interpreter call per
//! value). Plus the total `is_result` predicate and `get_error`'s string-payload
//! message (demo-extensions R1 / R2).

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let mut env = Environment::new();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), &mut env)?;
    }
    Ok(last)
}

fn values(src: &str) -> Vec<f64> {
    match eval(src).unwrap() {
        Value::Array(a) => {
            assert_eq!(a.rank(), 1, "flat 1-D result from {src}");
            a.data().to_vec()
        }
        other => panic!("expected an array, got {other:?}"),
    }
}

fn err_text(src: &str) -> String {
    match eval(src) {
        Ok(v) => panic!("expected an error from {src}, got {v:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn unpack_bf16_is_exact() {
    // bf16 bit patterns of [1, -1, 2, 0.5, 0, 50] (top 16 bits of the f32).
    let src = "b = reinterpret(pack([16256, 49024, 16384, 16128, 0, 16968], \"u16\"), \"bf16\")\n\
               unpack(b, \"bf16\")";
    assert_eq!(values(src), vec![1.0, -1.0, 2.0, 0.5, 0.0, 50.0]);
}

#[test]
fn unpack_round_trips_every_packable_dtype() {
    for dtype in [
        "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f32", "f64",
    ] {
        let vals = if dtype.starts_with('u') {
            "[0, 1, 7, 100]"
        } else {
            "[0, 1, -7, 100]"
        };
        let got = values(&format!("unpack(pack({vals}, \"{dtype}\"), \"{dtype}\")"));
        let want: Vec<f64> = vals
            .trim_matches(['[', ']'])
            .split(", ")
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(got, want, "{dtype}");
    }
}

#[test]
fn unpack_keeps_f16_specials() {
    // f16 bits: +inf 0x7c00, -inf 0xfc00, NaN 0x7e00, smallest subnormal 0x0001.
    let got =
        values("unpack(reinterpret(pack([31744, 64512, 32256, 1], \"u16\"), \"f16\"), \"f16\")");
    assert_eq!(got[0], f64::INFINITY);
    assert_eq!(got[1], f64::NEG_INFINITY);
    assert!(got[2].is_nan());
    assert_eq!(got[3], 2f64.powi(-24));
}

#[test]
fn unpack_rejects_a_ragged_byte_length() {
    let msg = err_text("unpack(pack([1, 2, 3], \"u8\"), \"u16\")");
    assert!(msg.contains("3 bytes") && msg.contains("u16"), "{msg}");
}

#[test]
fn unpack_of_a_large_buffer_is_one_native_pass() {
    let start = std::time::Instant::now();
    let got = values("unpack(pack(zeros([2000000]), \"f32\"), \"f32\")");
    assert_eq!(got.len(), 2_000_000);
    // Generous ceiling for a debug build: one interpreter call per value
    // would take minutes.
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
}

#[test]
fn is_result_is_total() {
    let one = |s: &str| values(&format!("[{s}]"))[0];
    assert_eq!(one("is_result(ok(1))"), 1.0);
    assert_eq!(one("is_result(err(\"boom\"))"), 1.0);
    assert_eq!(one("is_result(7)"), 0.0);
    assert_eq!(one("is_result(\"text\")"), 0.0);
}

#[test]
fn get_error_on_a_string_payload_points_at_err_message() {
    let msg = err_text("get_error(err(\"boom\"))");
    assert!(msg.contains("err_message(r)"), "{msg}");
    assert!(!msg.contains("Stage 6"), "no roadmap talk in errors: {msg}");
}
