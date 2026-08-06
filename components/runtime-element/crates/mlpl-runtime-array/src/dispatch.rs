//! Name -> kernel dispatch for the array builtins, split from the
//! `lib.rs` facade (facades carry no executable logic) into the
//! numeric/scan half and the shape/slice half.

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

use crate::{compute, reduce, shape, slice, transform};

type CallResult = Option<Result<DenseArray, RuntimeError>>;

pub(crate) fn try_call(name: &str, args: Vec<DenseArray>) -> CallResult {
    // Route by category up front (no args clone on the hot path).
    match name {
        "iota" | "range" | "linspace" | "cumprod" | "running_product" | "running_sum"
        | "reduce_add" | "reduce_mul" | "argmax" | "dot" | "matmul" => numeric_call(name, args),
        _ => structural_call(name, args),
    }
}

/// Generators, scans, reductions, and linear algebra.
fn numeric_call(name: &str, args: Vec<DenseArray>) -> CallResult {
    match name {
        // "iota"/"cumprod": DEPRECATED aliases of range/running_product.
        "iota" | "range" => Some(compute::iota(name, args)),
        "linspace" => Some(compute::linspace(name, args)),
        "cumprod" | "running_product" => Some(reduce::running_product(name, args)),
        "running_sum" => Some(reduce::running_sum(name, args)),
        "reduce_add" | "reduce_mul" => Some(reduce::reduce(name, args)),
        "argmax" => Some(reduce::argmax(name, args)),
        "dot" => Some(compute::dot(name, args)),
        "matmul" => Some(compute::matmul(name, args)),
        _ => None,
    }
}

/// Shape introspection and structural re-arrangement.
fn structural_call(name: &str, args: Vec<DenseArray>) -> CallResult {
    match name {
        "shape" => Some(shape::shape(name, args)),
        "rank" => Some(shape::rank(name, args)),
        "depth" => Some(shape::depth(name, args)),
        "size" => Some(shape::size(name, args)),
        "tally" => Some(shape::tally(name, args)),
        "flatten" => Some(shape::flatten(name, args)),
        "reshape" => Some(transform::reshape(name, args)),
        "transpose" => Some(transform::transpose(name, args)),
        "transpose_axes" => Some(transform::transpose_axes(name, args)),
        "grade_up" | "grade_down" => Some(slice::grade(name, args)),
        "compress" => Some(slice::compress(name, args)),
        "pareto_front" => Some(slice::pareto_front(name, args)),
        "patchify" => Some(slice::patchify(name, args)),
        "concat" if args.len() == 3 => Some(slice::concat(name, args)),
        "take" => Some(slice::take(name, args)),
        "rotate" => Some(transform::rotate(name, args)),
        _ => None,
    }
}
