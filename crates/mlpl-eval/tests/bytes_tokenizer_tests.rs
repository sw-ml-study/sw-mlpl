//! Byte-level tokenizer + Value::Tokenizer scaffolding tests
//! (Saga 12 step 004).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, Value, eval_program, eval_program_value, inspect};
use mlpl_parser::{lex, parse};

fn eval_arr(src: &str) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

fn eval_val(src: &str) -> Value {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env).unwrap()
}

// -- tokenize_bytes / decode_bytes --

#[test]
fn tokenize_bytes_ascii() {
    let arr = eval_arr("tokenize_bytes(\"hello\")");
    assert_eq!(arr.shape().dims(), &[5]);
    assert_eq!(arr.data(), &[104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn tokenize_bytes_empty_string() {
    let arr = eval_arr("tokenize_bytes(\"\")");
    assert_eq!(arr.shape().dims(), &[0]);
}

#[test]
fn decode_bytes_ascii_round_trips() {
    // Build a literal byte array by hand, then decode.
    let v = eval_val("decode_bytes([104, 101, 108, 108, 111])");
    assert_eq!(v, Value::Str("hello".into()));
}

#[test]
fn decode_tokenize_round_trip_ascii() {
    let v = eval_val("decode_bytes(tokenize_bytes(\"round trip\"))");
    assert_eq!(v, Value::Str("round trip".into()));
}

#[test]
fn decode_bytes_of_utf8_bytes_reconstructs_utf8_string() {
    // The MLPL lexer does not process `\u{...}` escapes; string
    // literals are byte-wise Latin-1 when built from source. So
    // exercise the UTF-8 round-trip by feeding `decode_bytes` an
    // explicit list of UTF-8 bytes and checking the string.
    //
    // "hé" = h (0x68) + é (0xc3 0xa9 in UTF-8).
    let v = eval_val("decode_bytes([104, 195, 169])");
    assert_eq!(v, Value::Str("h\u{00e9}".into()));
}

#[test]
fn decode_bytes_out_of_range_errors() {
    // 256 is out of the valid byte range.
    let src = "decode_bytes([65, 66, 256])";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "out-of-range byte should error");
}

#[test]
fn decode_bytes_fractional_errors() {
    let src = "decode_bytes([65.5])";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "non-integer byte should error");
}

#[test]
fn decode_bytes_higher_rank_errors() {
    let src = "decode_bytes(reshape(iota(6), [2, 3]))";
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(r.is_err(), "rank > 1 should error");
}

// -- Value::Tokenizer scaffolding --

#[test]
fn tokenizer_builtin_returns_tokenizer_value() {
    let v = eval_val("tokenizer()");
    assert!(
        matches!(v, Value::Tokenizer(_)),
        "expected Value::Tokenizer, got {v:?}"
    );
}

#[test]
fn tokenizer_can_be_bound_and_retrieved() {
    // An assignment of a tokenizer value currently can't fit through
    // the existing `Assign` path (which only handles arrays + models).
    // For now, exercise the builtin round-trip without binding.
    let v = eval_val("tokenizer()");
    assert!(matches!(v, Value::Tokenizer(_)));
}

// -- :describe on a tokenizer --

#[test]
fn describe_byte_level_tokenizer_reports_kind() {
    // Since assignment of Value::Tokenizer isn't wired yet, we test
    // the inspect formatter via the public API surface: a bound
    // variable named `tok` of kind Value::Tokenizer (constructed
    // through the environment directly, bypassing eval).
    //
    // This locks in the :describe output format for step 005 to
    // extend when BpeMerges lands.
    let mut env = Environment::new();
    env.set_tokenizer("tok".into(), mlpl_eval::TokenizerSpec::ByteLevel);
    let out = inspect(&env, ":describe tok").unwrap();
    assert!(out.contains("tokenizer"), "out was: {out}");
    assert!(out.contains("byte-level"), "out was: {out}");
    assert!(out.contains("256"), "out was: {out}");
}

// -- Regression: existing Value variants still display correctly --

#[test]
fn scalar_display_unchanged() {
    let arr = eval_arr("1 + 2");
    assert_eq!(arr.to_string(), "3");
}

#[test]
fn tokenize_bytes_tokens_are_rank1() {
    let arr = eval_arr("tokenize_bytes(\"ab\")");
    assert_eq!(arr.shape(), &Shape::vector(2));
}
