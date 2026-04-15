//! Tests for `apply_tokenizer` and `decode` (Saga 12 step 006).
//!
//! Round-trip invariant: `decode(tok, apply_tokenizer(tok, s)) == s`
//! for every UTF-8 string s, whether or not s was in the training
//! corpus.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval_val(src: &str) -> Value {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env).unwrap()
}

fn eval_arr(src: &str) -> mlpl_array::DenseArray {
    eval_val(src).as_array().unwrap().clone()
}

// -- ByteLevel tokenizer --

#[test]
fn apply_bytelevel_matches_tokenize_bytes() {
    let a = eval_arr("apply_tokenizer(tokenizer(), \"hello\")");
    let b = eval_arr("tokenize_bytes(\"hello\")");
    assert_eq!(a.data(), b.data());
    assert_eq!(a.shape(), b.shape());
}

#[test]
fn decode_bytelevel_matches_decode_bytes() {
    let a = eval_val("decode(tokenizer(), [104, 101, 108, 108, 111])");
    let b = eval_val("decode_bytes([104, 101, 108, 108, 111])");
    assert_eq!(a, b);
    assert_eq!(a, Value::Str("hello".into()));
}

#[test]
fn bytelevel_round_trip_ascii() {
    let v = eval_val("decode(tokenizer(), apply_tokenizer(tokenizer(), \"round trip\"))");
    assert_eq!(v, Value::Str("round trip".into()));
}

#[test]
fn bytelevel_round_trip_utf8() {
    let v = eval_val("decode(tokenizer(), apply_tokenizer(tokenizer(), \"h\u{00e9}llo\"))");
    assert_eq!(v, Value::Str("h\u{00e9}llo".into()));
}

// -- BPE tokenizer --

#[test]
fn apply_bpe_compresses_trained_pairs() {
    // Training on "abababab" -> first merge is (97, 98). Apply on
    // the training corpus should collapse 4 `ab` pairs into 4
    // merged tokens, so the result is shorter than the raw byte
    // tokenization.
    let src = "bpe = train_bpe(\"abababab\", 260, 0)\n\
               t = apply_tokenizer(bpe, \"abababab\")\n\
               shape(t)";
    let shape_arr = eval_arr(src);
    // 8 bytes collapse to at most 4 merged tokens after the first
    // merge; with up to 4 merges allowed (vocab 260) we expect
    // <= 4.
    assert!(
        shape_arr.data()[0] <= 4.0,
        "got shape {:?}",
        shape_arr.data()
    );
}

#[test]
fn apply_bpe_uses_trained_ids_above_256() {
    // After training, at least one merged id >= 256 should appear
    // in the apply output (since the merge fired on the training
    // corpus).
    let src = "bpe = train_bpe(\"abababab\", 260, 0)\n\
               apply_tokenizer(bpe, \"abababab\")";
    let arr = eval_arr(src);
    assert!(
        arr.data().iter().any(|&v| v >= 256.0),
        "expected a token id >= 256, got {:?}",
        arr.data()
    );
}

#[test]
fn bpe_round_trip_on_training_corpus() {
    let v = eval_val(
        "bpe = train_bpe(\"the quick brown fox jumps over the lazy dog\", 280, 3)\n\
         decode(bpe, apply_tokenizer(bpe, \"the quick brown fox jumps over the lazy dog\"))",
    );
    assert_eq!(
        v,
        Value::Str("the quick brown fox jumps over the lazy dog".into())
    );
}

#[test]
fn bpe_round_trip_on_unseen_text() {
    // BPE is byte-lossless: any byte sequence round-trips even if
    // it never appeared in the training corpus.
    let v = eval_val(
        "bpe = train_bpe(\"the quick brown fox\", 270, 1)\n\
         decode(bpe, apply_tokenizer(bpe, \"unseen text 12345\"))",
    );
    assert_eq!(v, Value::Str("unseen text 12345".into()));
}

#[test]
fn bpe_round_trip_utf8() {
    let v = eval_val(
        "bpe = train_bpe(\"caf\u{00e9} caf\u{00e9}\", 260, 0)\n\
         decode(bpe, apply_tokenizer(bpe, \"caf\u{00e9}\"))",
    );
    assert_eq!(v, Value::Str("caf\u{00e9}".into()));
}

#[test]
fn apply_bpe_token_ids_bounded_by_vocab_size() {
    let src = "bpe = train_bpe(\"abababab\", 260, 0)\n\
               apply_tokenizer(bpe, \"abababab\")";
    let arr = eval_arr(src);
    // Vocab is at most 260, so token ids must be < 260.
    assert!(
        arr.data().iter().all(|&v| v < 260.0),
        "got ids {:?}",
        arr.data()
    );
}

#[test]
fn apply_accepts_byte_array_input() {
    // Apply on an explicit byte array should match apply on the
    // equivalent string.
    let from_str = eval_arr("apply_tokenizer(tokenizer(), \"hello\")");
    let from_arr = eval_arr("apply_tokenizer(tokenizer(), [104, 101, 108, 108, 111])");
    assert_eq!(from_str.data(), from_arr.data());
}

#[test]
fn decode_bpe_of_single_byte_token_is_byte() {
    // A byte id (< 256) should decode to its raw byte.
    let v = eval_val(
        "bpe = train_bpe(\"abababab\", 260, 0)\n\
         decode(bpe, [97, 98])",
    );
    assert_eq!(v, Value::Str("ab".into()));
}
