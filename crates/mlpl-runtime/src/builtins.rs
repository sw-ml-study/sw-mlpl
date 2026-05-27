//! Built-in function implementations.

use mlpl_array::{DenseArray, Shape};

use crate::math_builtins;
use mlpl_runtime_core::error::RuntimeError;

/// Names dispatched by the `match` block at the bottom of
/// `call_builtin`. Excludes names handled by the per-module
/// `try_call` chain above (those are listed in each module's
/// own `NAMES` constant).
pub(crate) const LOCAL_NAMES: &[&str] = &[
    "iota",
    "shape",
    "rank",
    "reshape",
    "transpose",
    "reduce_add",
    "reduce_mul",
    "argmax",
    "dot",
    "matmul",
    "grid",
    "patchify",
    "take",
    // Note: `concat` is not listed here. The 3-arg
    // axis-aware form (Saga 29 step 005) is dispatched
    // below alongside the 2-arg legacy form registered in
    // `math_builtins::NAMES`. Both arities flow through this
    // file's `call_builtin` -> 3-arg falls through
    // `math_builtins::try_call` and hits the `"concat"` arm
    // in the match below.
];

macro_rules! check_arity {
    ($name:expr, $expected:expr, $args:expr) => {
        if $args.len() != $expected {
            return Err(RuntimeError::ArityMismatch {
                func: $name.into(),
                expected: $expected,
                got: $args.len(),
            });
        }
    };
}

/// Try every external per-module `try_call` dispatcher in
/// registration order; returns `Some(result)` for the first
/// module that recognizes `name`. Extracted from
/// [`call_builtin`] so the latter stays under the
/// Function-LOC budget as the dispatcher chain grows
/// (saga 33 step 033 added the UMAP entry, putting the
/// inline chain over the 50-LOC budget).
fn try_external_dispatchers(
    name: &str,
    args: &[DenseArray],
) -> Option<Result<DenseArray, RuntimeError>> {
    math_builtins::try_call(name, args.to_vec())
        .or_else(|| crate::conv_builtins::try_call(name, args.to_vec()))
        .or_else(|| crate::random_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_data::dataset_builtins::try_call(name, args.to_vec()))
        .or_else(|| crate::ml_builtins::try_call(name, args.to_vec()))
        .or_else(|| crate::ensemble_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_data::embedding_builtins::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_dim_reduction::tsne_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_dim_reduction::pca_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_umap::umap_graph::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_umap::umap_builtin::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_mds_rp::mds::try_call(name, args.to_vec()))
        .or_else(|| mlpl_runtime_mds_rp::random_projection::try_call(name, args.to_vec()))
}

/// Dispatch a built-in function call by name.
pub fn call_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if let Some(result) = try_external_dispatchers(name, &args) {
        return result;
    }
    match_local_builtin(name, args)
}

/// Match the local in-crate builtin names. Extracted from
/// [`call_builtin`] so the outer entry point stays under the
/// Function-LOC budget; the inner match grows over time as
/// new local arms are added.
fn match_local_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    match name {
        "iota" => builtin_iota(name, args),
        "shape" => builtin_shape(name, args),
        "rank" => builtin_rank(name, args),
        "reshape" => builtin_reshape(name, args),
        "transpose" => builtin_transpose(name, args),
        "reduce_add" | "reduce_mul" => builtin_reduce(name, args),
        "argmax" => builtin_argmax(name, args),
        "dot" => builtin_dot(name, args),
        "matmul" => builtin_matmul(name, args),
        "grid" => mlpl_runtime_data::grid_builtin::builtin_grid(name, args),
        "patchify" => builtin_patchify(name, args),
        "concat" => builtin_concat(name, args),
        "take" => builtin_take(name, args),
        _ => Err(RuntimeError::UnknownFunction(name.into())),
    }
}

fn builtin_dot(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 2, args);
    Ok(args[0].dot(&args[1])?)
}

fn builtin_matmul(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 2, args);
    Ok(args[0].matmul(&args[1])?)
}

fn builtin_iota(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    if args[0].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("expected scalar, got rank {}", args[0].rank()),
        });
    }
    let n = args[0].data()[0] as usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Ok(DenseArray::from_vec(data))
}

fn builtin_shape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    let dims: Vec<f64> = args[0].shape().dims().iter().map(|&d| d as f64).collect();
    Ok(DenseArray::from_vec(dims))
}

fn builtin_rank(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    Ok(DenseArray::from_scalar(args[0].rank() as f64))
}

fn builtin_reshape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 2, args);
    let shape_data = args[1].data();
    let dims: Vec<usize> = shape_data.iter().map(|&d| d as usize).collect();
    Ok(args[0].reshape(Shape::new(dims))?)
}

fn builtin_transpose(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    Ok(args[0].transpose())
}

fn builtin_argmax(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() == 1 {
        // Flat argmax over all elements, returned as a scalar index.
        let data = args[0].data();
        if data.is_empty() {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: "argmax of empty array".into(),
            });
        }
        let mut best_idx = 0usize;
        let mut best_val = data[0];
        for (i, &v) in data.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return Ok(DenseArray::from_scalar(best_idx as f64));
    }
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    let axis = args[1].data()[0] as usize;
    Ok(args[0].argmax_axis(axis)?)
}

fn builtin_reduce(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let got = args.len();
    if got != 1 && got != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got,
        });
    }
    let (identity, op): (f64, fn(f64, f64) -> f64) = match name {
        "reduce_add" => (0.0, |a, b| a + b),
        "reduce_mul" => (1.0, |a, b| a * b),
        _ => unreachable!(),
    };
    if got == 1 {
        return Ok(DenseArray::from_scalar(
            args[0].data().iter().copied().fold(identity, op),
        ));
    }
    if args[1].rank() != 0 {
        let reason = format!("axis must be scalar, got rank {}", args[1].rank());
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason,
        });
    }
    let axis = args[1].data()[0] as usize;
    Ok(args[0].reduce_axis(axis, identity, op)?)
}

fn builtin_patchify(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 2, args);
    let p = scalar_usize(name, &args[1], "patch_size")?;
    Ok(args[0].patchify(p)?)
}

fn builtin_concat(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 3, args);
    let axis = scalar_usize(name, &args[2], "axis")?;
    Ok(args[0].concat(&args[1], axis)?)
}

fn builtin_take(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 3, args);
    let axis = scalar_usize(name, &args[1], "axis")?;
    let idx = scalar_usize(name, &args[2], "idx")?;
    Ok(args[0].take(axis, idx)?)
}

fn scalar_usize(name: &str, arr: &DenseArray, what: &str) -> Result<usize, RuntimeError> {
    if arr.rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a scalar, got rank {}", arr.rank()),
        });
    }
    let v = arr.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a non-negative integer, got {v}"),
        });
    }
    Ok(v as usize)
}
