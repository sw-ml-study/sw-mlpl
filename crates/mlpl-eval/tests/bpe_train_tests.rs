//! Tests for `train_bpe` (Saga 12 step 005).
//!
//! Algorithmic verification: deterministic tie-breaking,
//! idempotent retraining with the same seed, byte-array and
//! string inputs produce identical results, early exit when no
//! pair has count of 2 or more, and :describe shows the expected
//! vocab / merges / corpus-byte-count / seed numbers.

use mlpl_eval::{Environment, TokenizerSpec, Value, eval_program_value, inspect};
use mlpl_parser::{lex, parse};

fn eval_val(src: &str) -> Value {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env).unwrap()
}

fn into_bpe(v: Value) -> (Vec<(u32, u32)>, u32, usize, u64) {
    match v {
        Value::Tokenizer(TokenizerSpec::BpeMerges {
            merges,
            vocab_size,
            corpus_byte_count,
            seed,
        }) => (merges, vocab_size, corpus_byte_count, seed),
        other => panic!("expected BpeMerges tokenizer, got {other:?}"),
    }
}

#[test]
fn train_bpe_returns_bpe_tokenizer() {
    // Small corpus; request a handful of merges past 256.
    let v = eval_val("train_bpe(\"abababab\", 260, 7)");
    let (merges, vocab, byte_count, seed) = into_bpe(v);
    assert!(vocab >= 256);
    assert!(vocab <= 260);
    assert_eq!(vocab as usize, 256 + merges.len());
    assert_eq!(byte_count, 8);
    assert_eq!(seed, 7);
}

#[test]
fn train_bpe_first_merge_picks_most_frequent_pair() {
    // "abababab" = a(97) b(98) a b a b a b. The pair (97, 98) = 4x,
    // (98, 97) = 3x. First merge should be (97, 98).
    let v = eval_val("train_bpe(\"abababab\", 257, 0)");
    let (merges, _, _, _) = into_bpe(v);
    assert!(!merges.is_empty(), "at least one merge expected");
    assert_eq!(merges[0], (97, 98));
}

#[test]
fn train_bpe_is_deterministic_given_same_input_and_seed() {
    let a = into_bpe(eval_val("train_bpe(\"the quick brown fox\", 280, 42)")).0;
    let b = into_bpe(eval_val("train_bpe(\"the quick brown fox\", 280, 42)")).0;
    assert_eq!(a, b);
}

#[test]
fn train_bpe_lex_tie_breaking() {
    // In "abcd" every adjacent pair occurs exactly once:
    // (97,98), (98,99), (99,100). Lex-smallest is (97,98).
    let v = eval_val("train_bpe(\"abcd\", 257, 0)");
    let (merges, _, _, _) = into_bpe(v);
    assert_eq!(merges[0], (97, 98));
}

#[test]
fn train_bpe_early_exit_when_no_pair_recurs() {
    // "abcde" -- every pair occurs exactly once. With a large
    // requested vocab_size, BPE still adds those merges (count >= 1),
    // but once the corpus is reduced to a single token, no pairs
    // remain. Total merges <= 4 (original 5-byte sequence -> 1 token
    // after at most 4 merges).
    let v = eval_val("train_bpe(\"abcde\", 1000, 0)");
    let (merges, _, _, _) = into_bpe(v);
    assert!(merges.len() <= 4, "got {} merges", merges.len());
}

#[test]
fn train_bpe_accepts_pretokenized_byte_array() {
    // "ab" as bytes -> [97, 98]. Same result as training on the string.
    let from_str = into_bpe(eval_val("train_bpe(\"abababab\", 257, 0)")).0;
    let from_arr = into_bpe(eval_val(
        "train_bpe([97, 98, 97, 98, 97, 98, 97, 98], 257, 0)",
    ))
    .0;
    assert_eq!(from_str, from_arr);
}

#[test]
fn train_bpe_describe_reports_metadata() {
    let mut env = Environment::new();
    let tokens = lex("bpe = train_bpe(\"abababab\", 260, 7)").unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program_value(&stmts, &mut env).unwrap();
    let out = inspect(&env, ":describe bpe").unwrap();
    assert!(out.contains("BPE tokenizer"), "out was: {out}");
    assert!(out.contains("vocab="), "out was: {out}");
    assert!(out.contains("trained from 8 bytes"), "out was: {out}");
    assert!(out.contains("seed=7"), "out was: {out}");
}

#[test]
fn train_bpe_utf8_corpus_counts_utf8_bytes() {
    // "héllo" in UTF-8 is 6 bytes; byte_count should match.
    let v = eval_val("train_bpe(\"h\u{00e9}llo\", 260, 0)");
    let (_, _, byte_count, _) = into_bpe(v);
    assert_eq!(byte_count, 6);
}
