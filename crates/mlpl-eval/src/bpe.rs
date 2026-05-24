//! BPE eval-side adapters. Saga 33 step 018: the pure
//! train + apply_trained + decode_token algorithm moved to
//! `mlpl-bpe-core`; this file keeps the env / Value /
//! tokenizer-registry plumbing.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::tokenizer::TokenizerSpec;
use crate::value::Value;

pub(crate) use mlpl_bpe_core::train;

/// Accept either a `Value::Str` (treated as UTF-8 bytes) or a
/// rank-1 `Value::Array` of byte-indices 0..=255. Returns the
/// byte sequence; other variants error.
pub(crate) fn corpus_to_bytes(v: Value) -> Result<Vec<u8>, EvalError> {
    match v {
        Value::Str(s) => Ok(s.into_bytes()),
        Value::Array(a) => byte_array_to_bytes(&a),
        _ => Err(EvalError::Unsupported(
            "train_bpe: corpus must be a string or a rank-1 byte array".into(),
        )),
    }
}

fn byte_array_to_bytes(a: &DenseArray) -> Result<Vec<u8>, EvalError> {
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

/// `apply_tokenizer(tok, text)` dispatch helper.
pub(crate) fn dispatch_apply(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    arity(args, 2, "apply_tokenizer")?;
    let tok = resolve_tokenizer(&args[0], env, trace)?;
    let text_val = crate::eval::eval_expr(&args[1], env, trace)?;
    let bytes = corpus_to_bytes(text_val)?;
    let ids: Vec<f64> = match tok {
        TokenizerSpec::ByteLevel => bytes.iter().map(|&b| f64::from(b)).collect(),
        TokenizerSpec::BpeMerges { merges, .. } => mlpl_bpe_core::apply_trained(&bytes, &merges)
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    Ok(Value::Array(DenseArray::new(
        Shape::vector(ids.len()),
        ids,
    )?))
}

/// `decode(tok, tokens)` dispatch helper.
pub(crate) fn dispatch_decode(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    arity(args, 2, "decode")?;
    let tok = resolve_tokenizer(&args[0], env, trace)?;
    let arr = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    match tok {
        TokenizerSpec::ByteLevel => crate::tokenizer::eval_decode_bytes(&arr),
        TokenizerSpec::BpeMerges { merges, .. } => decode_bpe_ids(&arr, &merges),
    }
}

fn arity(args: &[Expr], expected: usize, func: &str) -> Result<(), EvalError> {
    if args.len() == expected {
        return Ok(());
    }
    Err(EvalError::BadArity {
        func: func.into(),
        expected,
        got: args.len(),
    })
}

/// Resolve the first-arg tokenizer slot. An `Ident` that
/// names a bound tokenizer wins first; anything else is
/// evaluated and required to produce `Value::Tokenizer`.
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
        mlpl_bpe_core::decode_token(v as u32, merges, &mut bytes);
    }
    match String::from_utf8(bytes) {
        Ok(s) => Ok(Value::Str(s)),
        Err(e) => Ok(Value::Str(
            String::from_utf8_lossy(&e.into_bytes()).into_owned(),
        )),
    }
}
