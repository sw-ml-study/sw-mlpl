//! Higher-level ML built-in functions (softmax, one_hot,
//! sinusoidal_encoding, cross_entropy, perplexity) extracted
//! from mlpl-runtime.

mod activation;
mod encoding;
mod losses;

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &[
    "softmax",
    "one_hot",
    "sinusoidal_encoding",
    "cross_entropy",
    "perplexity",
];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "softmax" => Some(activation::softmax(name, args)),
        "one_hot" => Some(encoding::one_hot(name, args)),
        "sinusoidal_encoding" => Some(encoding::sinusoidal(name, args)),
        "cross_entropy" => Some(losses::cross_entropy(name, args)),
        "perplexity" => Some(losses::perplexity(name, args)),
        _ => None,
    }
}

pub(crate) fn arity_err(name: &str, expected: usize, got: usize) -> RuntimeError {
    RuntimeError::ArityMismatch {
        func: name.into(),
        expected,
        got,
    }
}
