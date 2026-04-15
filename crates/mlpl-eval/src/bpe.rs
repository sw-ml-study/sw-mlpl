//! BPE training algorithm (Saga 12 step 005).
//!
//! Split out of `tokenizer.rs` so both modules stay under the
//! sw-checklist function-count budget. Exposes three helpers:
//!
//! - `corpus_to_bytes` -- normalize a `Value::Str` or rank-1 byte
//!   array into a `Vec<u8>` for training input.
//! - `train` -- byte-level BPE. Returns the ordered merge list;
//!   caller wraps it in `TokenizerSpec::BpeMerges`.
//! - Two private helpers (`pick_merge_pair`, `apply_merge`) sit
//!   behind `train`.

use std::collections::HashMap;

use crate::error::EvalError;
use crate::value::Value;

/// Accept either a `Value::Str` (treated as UTF-8 bytes) or a
/// rank-1 `Value::Array` of byte-indices 0..=255. Returns the
/// byte sequence; other variants error.
pub(crate) fn corpus_to_bytes(v: Value) -> Result<Vec<u8>, EvalError> {
    match v {
        Value::Str(s) => Ok(s.into_bytes()),
        Value::Array(a) => {
            if a.rank() > 1 {
                return Err(EvalError::Unsupported(format!(
                    "train_bpe: corpus array must be rank <= 1, got rank {}",
                    a.rank()
                )));
            }
            let mut out = Vec::with_capacity(a.data().len());
            for (i, &v) in a.data().iter().enumerate() {
                if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
                    return Err(EvalError::Unsupported(format!(
                        "train_bpe: corpus cell {i} = {v} is not an integer in 0..=255"
                    )));
                }
                out.push(v as u8);
            }
            Ok(out)
        }
        _ => Err(EvalError::Unsupported(
            "train_bpe: corpus must be a string or a rank-1 byte array".into(),
        )),
    }
}

/// Byte-level BPE. Walks the corpus as a sequence of u32 tokens
/// (starting as the 256 byte ids), and at each step: counts
/// adjacent pairs, picks the most frequent pair with lex-smallest
/// `(left_id, right_id)` tie-breaking, adds it as a new token,
/// rewrites the sequence, and repeats until `vocab_size` is
/// reached or no adjacent pairs remain.
pub(crate) fn train(corpus: &[u8], vocab_size: u32) -> Vec<(u32, u32)> {
    let mut tokens: Vec<u32> = corpus.iter().map(|&b| u32::from(b)).collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut next_id: u32 = 256;
    while next_id < vocab_size && tokens.len() >= 2 {
        let Some(pair) = pick_merge_pair(&tokens) else {
            break;
        };
        merges.push(pair);
        tokens = apply_merge(&tokens, pair, next_id);
        next_id += 1;
    }
    merges
}

/// Count adjacent pairs in `tokens` and return the winning pair
/// (highest count, lex-smallest on ties) with count >= 1. Returns
/// `None` if no adjacent pairs exist.
fn pick_merge_pair(tokens: &[u32]) -> Option<(u32, u32)> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for w in tokens.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|(pa, ca), (pb, cb)| {
            // Higher count wins; on tie, lex-smallest pair wins.
            ca.cmp(cb).then_with(|| pb.cmp(pa))
        })
        .map(|(p, _)| p)
}

/// Produce a new token sequence with every occurrence of `pair`
/// replaced by `new_id`. Left-to-right greedy (standard BPE apply
/// order).
fn apply_merge(tokens: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(tokens[i]);
            i += 1;
        }
    }
    out
}
