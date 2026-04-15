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

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::tokenizer::TokenizerSpec;
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

/// `apply_tokenizer(tok, text)` dispatch helper. Shared between
/// ByteLevel (identity map on bytes) and BpeMerges (`apply_trained`).
pub(crate) fn dispatch_apply(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "apply_tokenizer".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let tok = resolve_tokenizer(&args[0], env, trace)?;
    let text_val = crate::eval::eval_expr(&args[1], env, trace)?;
    let bytes = corpus_to_bytes(text_val)?;
    let ids: Vec<f64> = match tok {
        TokenizerSpec::ByteLevel => bytes.iter().map(|&b| f64::from(b)).collect(),
        TokenizerSpec::BpeMerges { merges, .. } => apply_trained(&bytes, &merges)
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    Ok(Value::Array(DenseArray::new(
        Shape::vector(ids.len()),
        ids,
    )?))
}

/// `decode(tok, tokens)` dispatch helper. ByteLevel delegates to
/// `decode_bytes`; BpeMerges recursively expands each token id
/// through the merges table.
pub(crate) fn dispatch_decode(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "decode".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let tok = resolve_tokenizer(&args[0], env, trace)?;
    let arr = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    match tok {
        TokenizerSpec::ByteLevel => crate::tokenizer::eval_decode_bytes(&arr),
        TokenizerSpec::BpeMerges { merges, .. } => decode_bpe_ids(&arr, &merges),
    }
}

/// Resolve the first-arg tokenizer slot. An `Ident` that names a
/// bound tokenizer wins first; anything else is evaluated and
/// required to produce `Value::Tokenizer`.
fn resolve_tokenizer(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<TokenizerSpec, EvalError> {
    if let Expr::Ident(name, _) = expr
        && let Some(tok) = env.get_tokenizer(name)
    {
        return Ok(tok.clone());
    }
    match crate::eval::eval_expr(expr, env, trace)? {
        Value::Tokenizer(t) => Ok(t),
        _ => Err(EvalError::Unsupported(
            "expected a tokenizer (use tokenizer() or train_bpe(...))".into(),
        )),
    }
}

fn decode_bpe_ids(arr: &DenseArray, merges: &[(u32, u32)]) -> Result<Value, EvalError> {
    if arr.rank() > 1 {
        return Err(EvalError::Unsupported(format!(
            "decode: expected rank <= 1 token array, got rank {}",
            arr.rank()
        )));
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(arr.data().len());
    for (i, &v) in arr.data().iter().enumerate() {
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EvalError::Unsupported(format!(
                "decode: cell {i} = {v} is not a non-negative integer token id"
            )));
        }
        decode_token(v as u32, merges, &mut bytes);
    }
    match String::from_utf8(bytes) {
        Ok(s) => Ok(Value::Str(s)),
        Err(e) => Ok(Value::Str(
            String::from_utf8_lossy(&e.into_bytes()).into_owned(),
        )),
    }
}

/// Apply a trained merge list to `bytes`, returning the compressed
/// token sequence. Walks merges in training order; each merge is
/// applied with `apply_merge` (same greedy left-to-right routine
/// used during training). Saga 12 step 006.
pub(crate) fn apply_trained(bytes: &[u8], merges: &[(u32, u32)]) -> Vec<u32> {
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    for (i, &pair) in merges.iter().enumerate() {
        let new_id = 256 + u32::try_from(i).unwrap_or(u32::MAX - 256);
        tokens = apply_merge(&tokens, pair, new_id);
    }
    tokens
}

/// Recursively expand a single BPE token id back into its byte
/// sequence. Byte ids (< 256) decode to themselves; merged ids
/// (>= 256) recursively expand their `(left, right)` pair. Saga
/// 12 step 006.
pub(crate) fn decode_token(id: u32, merges: &[(u32, u32)], out: &mut Vec<u8>) {
    if id < 256 {
        out.push(id as u8);
        return;
    }
    let merge_idx = (id - 256) as usize;
    if merge_idx >= merges.len() {
        // Unknown merged id -- skip; defensive. Valid trained
        // outputs never produce ids outside this range.
        return;
    }
    let (l, r) = merges[merge_idx];
    decode_token(l, merges, out);
    decode_token(r, merges, out);
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
