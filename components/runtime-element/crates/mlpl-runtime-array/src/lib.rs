//! Array shape / reduce / linalg / slice built-in functions
//! extracted from mlpl-runtime.

mod compute;
mod dispatch;
mod reduce;
mod shape;
mod slice;
mod transform;

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &[
    "range",
    "linspace",
    "running_product",
    "running_sum",
    "shape",
    "rank",
    "depth",
    "size",
    "tally",
    "flatten",
    "grade_up",
    "grade_down",
    "compress",
    "pareto_front",
    "reshape",
    "transpose",
    "reduce_add",
    "reduce_mul",
    "argmax",
    "dot",
    "matmul",
    "patchify",
    "take",
    "rotate",
];
// Note: `concat` (3-arg axis-aware form) is dispatched here but
// not listed in NAMES because the 2-arg legacy form is registered
// in mlpl-runtime-math's NAMES; listing both would duplicate.

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    dispatch::try_call(name, args)
}

pub(crate) fn arity_err(name: &str, expected: usize, got: usize) -> RuntimeError {
    RuntimeError::ArityMismatch {
        func: name.into(),
        expected,
        got,
    }
}
