//! Argument parsing for the Engram primitive builtins (split from
//! `fncall_engram.rs`, saga E3): array/scalar/integral coercions and
//! the five-argument `ngram_hash` input bundle.

use mlpl_array::DenseArray;
use mlpl_engram_core::HashSpec;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;

/// `ngram_hash(ids, orders, heads, slots, seed)` inputs: validated
/// ids plus the assembled [`HashSpec`].
pub(crate) fn hash_inputs(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(Vec<u64>, HashSpec), EvalError> {
    let ids = array_arg(&args[0], env, trace)?;
    let orders = array_arg(&args[1], env, trace)?;
    let spec = HashSpec {
        ngram_orders: integral_vec(&orders, "ngram_hash: orders")?
            .into_iter()
            .map(|v| v as usize)
            .collect(),
        heads_per_ngram: usize_arg(&args[2], env, trace, "ngram_hash: heads")?,
        slots_per_head: usize_arg(&args[3], env, trace, "ngram_hash: slots")?,
        seed: usize_arg(&args[4], env, trace, "ngram_hash: seed")? as u64,
    };
    Ok((integral_vec(&ids, "ngram_hash: ids")?, spec))
}

/// Evaluate an argument to an array.
pub(crate) fn array_arg(
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    eval_expr(arg, env, trace)?.into_array()
}

/// Evaluate an argument to a non-negative integer scalar.
pub(crate) fn usize_arg(
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    what: &str,
) -> Result<usize, EvalError> {
    let arr = array_arg(arg, env, trace)?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!("{what} must be a scalar")));
    }
    let v = arr.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{what} must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}

/// Every element as an exact non-negative integer (u64), or a loud
/// error naming the argument -- fractional or negative values can
/// never be token ids or table indices.
pub(crate) fn integral_vec(arr: &DenseArray, what: &str) -> Result<Vec<u64>, EvalError> {
    arr.data()
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                Err(EvalError::Unsupported(format!(
                    "{what} must be non-negative integers, got {v}"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect()
}
