//! Array shape / reduce / linalg / slice built-in functions
//! extracted from mlpl-runtime.

mod compute;
mod reduce;
mod shape;
mod slice;

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &[
    "iota",
    "range",
    "shape",
    "rank",
    "reshape",
    "transpose",
    "reduce_add",
    "reduce_mul",
    "argmax",
    "dot",
    "matmul",
    "patchify",
    "take",
];
// Note: `concat` (3-arg axis-aware form) is dispatched here but
// not listed in NAMES because the 2-arg legacy form is registered
// in mlpl-runtime-math's NAMES; listing both would duplicate.

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "iota" | "range" => Some(compute::iota(name, args)),
        "shape" => Some(shape::shape(name, args)),
        "rank" => Some(shape::rank(name, args)),
        "reshape" => Some(shape::reshape(name, args)),
        "transpose" => Some(shape::transpose(name, args)),
        "reduce_add" | "reduce_mul" => Some(reduce::reduce(name, args)),
        "argmax" => Some(reduce::argmax(name, args)),
        "dot" => Some(compute::dot(name, args)),
        "matmul" => Some(compute::matmul(name, args)),
        "patchify" => Some(slice::patchify(name, args)),
        "concat" if args.len() == 3 => Some(slice::concat(name, args)),
        "take" => Some(slice::take(name, args)),
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
