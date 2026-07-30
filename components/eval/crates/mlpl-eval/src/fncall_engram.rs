//! Engram primitive builtins (saga E1 step 2): `ngram_hash` and
//! `gather_rows`. The hash delegates to the `mlpl-engram-core`
//! REFERENCE, so CPU output equals the frozen cross-backend fixture
//! by construction; `gather_rows` is the general flattened-table
//! row gather every Engram lookup composes with.

use mlpl_array::{DenseArray, Shape};
use mlpl_engram_core::HashSpec;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "ngram_hash" => Some(eval_ngram_hash(args, env, trace)),
        "gather_rows" => Some(eval_gather_rows(args, env, trace)),
        _ => None,
    }
}

/// `ngram_hash(ids, orders, heads, slots, seed)` -> rank-3
/// `[T, order, head]` LOCAL slot indices, computed by the frozen
/// exact-arithmetic reference (docs/engram-sagas-plan.md D4).
fn eval_ngram_hash(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 5 {
        return Err(EvalError::BadArity {
            func: "ngram_hash".into(),
            expected: 5,
            got: args.len(),
        });
    }
    let ids = array_arg(&args[0], env, trace)?;
    let orders = array_arg(&args[1], env, trace)?;
    let heads = usize_arg(&args[2], env, trace, "ngram_hash: heads")?;
    let slots = usize_arg(&args[3], env, trace, "ngram_hash: slots")?;
    let seed = usize_arg(&args[4], env, trace, "ngram_hash: seed")? as u64;
    let spec = HashSpec {
        ngram_orders: integral_vec(&orders, "ngram_hash: orders")?
            .into_iter()
            .map(|v| v as usize)
            .collect(),
        heads_per_ngram: heads,
        slots_per_head: slots,
        seed,
    };
    let id_vals = integral_vec(&ids, "ngram_hash: ids")?;
    let hashes = mlpl_engram_core::ngram_hashes(&id_vals, &spec)
        .map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let flat: Vec<f64> = hashes
        .iter()
        .flatten()
        .flatten()
        .map(|&slot| slot as f64)
        .collect();
    let shape = Shape::new(vec![id_vals.len(), spec.ngram_orders.len(), heads]);
    Ok(Value::Array(DenseArray::new(shape, flat)?))
}

/// `gather_rows(table, indices)` -> rows of a rank-2 table selected
/// by an any-rank index array; output shape is
/// `indices.shape + [row_dim]`.
fn eval_gather_rows(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "gather_rows".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let table = array_arg(&args[0], env, trace)?;
    let indices = array_arg(&args[1], env, trace)?;
    if table.rank() != 2 {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: table must be rank 2, got rank {}",
            table.rank()
        )));
    }
    let (rows, dim) = (table.shape().dims()[0], table.shape().dims()[1]);
    let idx = integral_vec(&indices, "gather_rows: indices")?;
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &row in &idx {
        let row = row as usize;
        if row >= rows {
            return Err(EvalError::Unsupported(format!(
                "gather_rows: index {row} out of range for {rows} rows"
            )));
        }
        out.extend_from_slice(&table.data()[row * dim..(row + 1) * dim]);
    }
    let mut dims = indices.shape().dims().to_vec();
    dims.push(dim);
    Ok(Value::Array(DenseArray::new(Shape::new(dims), out)?))
}

/// Evaluate an argument to an array.
fn array_arg(
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    eval_expr(arg, env, trace)?.into_array()
}

/// Evaluate an argument to a non-negative integer scalar.
fn usize_arg(
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
fn integral_vec(arr: &DenseArray, what: &str) -> Result<Vec<u64>, EvalError> {
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
