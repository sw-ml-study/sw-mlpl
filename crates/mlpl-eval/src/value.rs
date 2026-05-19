//! Runtime value type for MLPL.

use std::collections::BTreeMap;
use std::fmt;

use mlpl_array::DenseArray;

use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::tokenizer::TokenizerSpec;

/// A runtime value: an array, a string, a model, a tokenizer,
/// or a peer-resident device tensor reference.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A dense array (scalar, vector, matrix, ...).
    Array(DenseArray),
    /// A string (introduced for diagram type names, LLM prompts, etc.).
    Str(String),
    /// A model: a callable layer (or composition) with attached
    /// parameters held in the environment. Saga 11.
    Model(ModelSpec),
    /// A tokenizer (Saga 12 step 004). Sibling to `Model` -- holds
    /// the tokenization strategy as data.
    Tokenizer(TokenizerSpec),
    /// A reference to a builtin or operator. Produced by the
    /// `:foo` / `:+` / `:max` syntax. Higher-order builtins
    /// like `reduce` and `map` accept this in place of a
    /// (future) function value -- it is the canonical first-
    /// class-ish op handle in v0.19.
    BuiltinRef {
        /// The builtin name without the leading `:`. Examples:
        /// `"add"`, `"+"`, `"max"`, `"sigmoid"`.
        name: String,
    },
    /// Saga R1 step 002: peer-resident tensor reference. The bytes
    /// live on the peer named by `peer`; the orchestrator only holds
    /// the handle + shape + device metadata. Touching it from a CPU
    /// op errors strict-fault; explicit `to_device('cpu', x)` is the
    /// only way to materialize the bytes back as a `Value::Array`.
    DeviceTensor {
        /// Peer URL the bytes live on.
        peer: String,
        /// Opaque handle issued by the peer.
        handle: String,
        /// Tensor shape (orchestrator-side cached so callers can
        /// `shape(x)` without a network round-trip).
        shape: Vec<usize>,
        /// Device name (e.g., `"mlx"`, future `"cuda"`).
        device: String,
    },
    /// Saga 29 step 001: structured record value. Field names map
    /// to arbitrary values (including nested records). BTreeMap so
    /// the key ordering is deterministic for display, trace
    /// serialization, and equality testing.
    Record {
        /// Field name -> value, sorted by name.
        fields: BTreeMap<String, Value>,
    },
    /// Saga 29 step 002: list of strings. Sibling to `Array`
    /// (the numeric DenseArray path). Produced by `[...]` literals
    /// whose elements all evaluate to `Value::Str`, and consumed
    /// by `list_len(xs)` plus future string-indexing builtins.
    /// Mixed-kind `[...]` literals raise
    /// `EvalError::MixedArrayLitElements`.
    StrList {
        /// The items in source order.
        items: Vec<String>,
    },
}

impl Value {
    /// Extract the inner array, returning an error if this is a
    /// string, model, tokenizer, or device tensor.
    pub fn into_array(self) -> Result<DenseArray, EvalError> {
        match self {
            Self::Array(a) => Ok(a),
            Self::DeviceTensor { peer, device, .. } => {
                Err(EvalError::DeviceTensorFault { peer, device })
            }
            _ => Err(EvalError::ExpectedArray),
        }
    }

    /// Borrow the inner array, returning an error if this is a
    /// string, model, tokenizer, or device tensor.
    pub fn as_array(&self) -> Result<&DenseArray, EvalError> {
        match self {
            Self::Array(a) => Ok(a),
            Self::DeviceTensor { peer, device, .. } => Err(EvalError::DeviceTensorFault {
                peer: peer.clone(),
                device: device.clone(),
            }),
            _ => Err(EvalError::ExpectedArray),
        }
    }
}

impl From<DenseArray> for Value {
    fn from(a: DenseArray) -> Self {
        Self::Array(a)
    }
}

/// Saga 29 step 001: human-readable variant name for tutoring
/// error messages (e.g., `FieldOnNonRecord`). Stable strings;
/// callers may match on them in tests.
pub fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Array(_) => "array",
        Value::Str(_) => "string",
        Value::Model(_) => "model",
        Value::Tokenizer(_) => "tokenizer",
        Value::BuiltinRef { .. } => "builtin-ref",
        Value::DeviceTensor { .. } => "device-tensor",
        Value::Record { .. } => "record",
        Value::StrList { .. } => "string-list",
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Array(a) => write!(f, "{a}"),
            Self::Str(s) => write!(f, "{s}"),
            Self::Model(_) => write!(f, "<model>"),
            Self::Tokenizer(t) => write!(f, "<tokenizer: {}>", t.describe()),
            Self::BuiltinRef { name } => write!(f, ":{name}"),
            Self::DeviceTensor {
                peer,
                device,
                shape,
                ..
            } => write!(f, "<tensor on {peer}:{device}; shape={shape:?}>"),
            Self::Record { fields } => {
                write!(f, "{{")?;
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {value}")?;
                }
                write!(f, "}}")
            }
            Self::StrList { items } => {
                // Saga 29 step 009 follow-up: truncate the
                // display when there are too many items so
                // long pets_tiny.names (200 strings) doesn't
                // dump several KB of text into the REPL on
                // every print.
                const LIMIT: usize = 8;
                write!(f, "[")?;
                let take = items.len().min(LIMIT);
                for (i, item) in items.iter().take(take).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\"", item.replace('\\', "\\\\").replace('"', "\\\""))?;
                }
                if items.len() > LIMIT {
                    write!(f, ", ... +{} more", items.len() - LIMIT)?;
                }
                write!(f, "]")
            }
        }
    }
}
