//! Fixed-width unsigned bit operations over exact-f64 integers
//! (docs/bit-ops-design.md): band/bor/bxor/bnot/popcount (logic)
//! plus shl/shr/bmask/bits/from_bits (shift + views). Pure and
//! universal; the domain is non-negative integers in 0..2^53.

mod logic;

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &["band", "bor", "bxor", "bnot", "popcount"];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    logic::try_call(name, args)
}

pub(crate) fn arity_err(name: &str, expected: usize, got: usize) -> RuntimeError {
    RuntimeError::ArityMismatch {
        func: name.into(),
        expected,
        got,
    }
}
