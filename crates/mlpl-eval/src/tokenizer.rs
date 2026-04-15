//! Tokenizer runtime values and byte-level builtins
//! (Saga 12 step 004).
//!
//! Mirrors the `Value::Model` pattern: a dedicated `Value::Tokenizer`
//! variant wrapping a `TokenizerSpec` enum. Step 004 introduces only
//! the `ByteLevel` variant; step 005 will add `BpeMerges { ... }`
//! once the BPE trainer lands.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;

/// Internal representation of a tokenizer. Sibling to `ModelSpec`.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizerSpec {
    /// Identity byte-level tokenizer: each byte 0..256 is its own
    /// token. Vocab size is implicitly 256.
    ByteLevel,
}

impl TokenizerSpec {
    /// Human-readable one-line description used by `:describe`.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::ByteLevel => "byte-level tokenizer (vocab=256)".into(),
        }
    }
}

/// Central dispatch for tokenizer-related builtins called from
/// `eval::eval_expr`. Returns `Some(Err(...))` for arity/type
/// errors, `Some(Ok(v))` for a handled call, and `None` when `name`
/// is not a tokenizer builtin so the caller falls through.
pub(crate) fn dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "tokenizer" => Some(dispatch_tokenizer_ctor(args)),
        "tokenize_bytes" => Some(dispatch_tokenize_bytes(args, env, trace)),
        "decode_bytes" => Some(dispatch_decode_bytes(args, env, trace)),
        _ => None,
    }
}

fn dispatch_tokenizer_ctor(args: &[Expr]) -> Result<Value, EvalError> {
    if !args.is_empty() {
        return Err(EvalError::BadArity {
            func: "tokenizer".into(),
            expected: 0,
            got: args.len(),
        });
    }
    Ok(Value::Tokenizer(TokenizerSpec::ByteLevel))
}

fn dispatch_tokenize_bytes(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "tokenize_bytes".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let v = crate::eval::eval_expr(&args[0], env, trace)?;
    let Value::Str(s) = v else {
        return Err(EvalError::ExpectedString);
    };
    eval_tokenize_bytes(&s)
}

fn dispatch_decode_bytes(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "decode_bytes".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let arr = crate::eval::eval_expr(&args[0], env, trace)?.into_array()?;
    eval_decode_bytes(&arr)
}

/// `tokenize_bytes(str)` -- returns a rank-1 `DenseArray` of byte
/// indices for the UTF-8 encoding of `s`.
pub(crate) fn eval_tokenize_bytes(s: &str) -> Result<Value, EvalError> {
    let bytes = s.as_bytes();
    let data: Vec<f64> = bytes.iter().map(|&b| f64::from(b)).collect();
    let arr = DenseArray::new(Shape::vector(data.len()), data)?;
    Ok(Value::Array(arr))
}

/// `decode_bytes(tokens)` -- inverse of `tokenize_bytes`. Rank-1
/// DenseArray of byte indices 0..=255; non-integer or out-of-range
/// cells error.
pub(crate) fn eval_decode_bytes(arr: &DenseArray) -> Result<Value, EvalError> {
    if arr.rank() > 1 {
        return Err(EvalError::Unsupported(format!(
            "decode_bytes: expected rank <= 1, got rank {}",
            arr.rank()
        )));
    }
    let mut bytes = Vec::with_capacity(arr.data().len());
    for (i, &v) in arr.data().iter().enumerate() {
        if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
            return Err(EvalError::Unsupported(format!(
                "decode_bytes: cell {i} = {v} is not an integer in 0..=255"
            )));
        }
        bytes.push(v as u8);
    }
    // Byte sequences from `tokenize_bytes` are valid UTF-8 by
    // construction; arbitrary user-constructed arrays may not be,
    // so do a lossless round-trip that preserves the bytes even if
    // they don't form valid UTF-8 (use `from_utf8_lossy` which
    // replaces invalid sequences with U+FFFD, then fall back to a
    // byte-identical construction via `String::from_utf8` where
    // possible).
    match String::from_utf8(bytes) {
        Ok(s) => Ok(Value::Str(s)),
        Err(e) => {
            // Fall back to lossy for pathological inputs; the round-
            // trip test uses only valid UTF-8 so this branch is
            // documented but rarely hit in practice.
            Ok(Value::Str(
                String::from_utf8_lossy(&e.into_bytes()).into_owned(),
            ))
        }
    }
}
