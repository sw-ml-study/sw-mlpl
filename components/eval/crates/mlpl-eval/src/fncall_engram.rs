//! Engram primitive builtins (saga E1 step 2): `ngram_hash` and
//! `gather_rows`. The hash delegates to the `mlpl-engram-core`
//! REFERENCE, so CPU output equals the frozen cross-backend fixture
//! by construction; `gather_rows` is the general flattened-table
//! row gather every Engram lookup composes with. Argument parsing
//! lives in `fncall_engram_args` (saga E3 split).

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_engram_args::{array_arg, hash_inputs, integral_vec};
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
    let (id_vals, spec) = hash_inputs(args, env, trace)?;
    let hashes = mlpl_engram_core::ngram_hashes(&id_vals, &spec)
        .map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let flat: Vec<f64> = hashes
        .iter()
        .flatten()
        .flatten()
        .map(|&s| s as f64)
        .collect();
    let (t, o, h) = (id_vals.len(), spec.ngram_orders.len(), spec.heads_per_ngram);
    let shape = Shape::new(vec![t, o, h]);
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
    crate::grad::arity_check(args, 2, "gather_rows")?;
    let table = array_arg(&args[0], env, trace)?;
    let indices = array_arg(&args[1], env, trace)?;
    if table.rank() != 2 {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: table must be rank 2, got rank {}",
            table.rank()
        )));
    }
    let idx = integral_vec(&indices, "gather_rows: indices")?;
    let out = gathered_rows(&table, &idx)?;
    let mut dims = indices.shape().dims().to_vec();
    dims.push(table.shape().dims()[1]);
    Ok(Value::Array(DenseArray::new(Shape::new(dims), out)?))
}

/// Copy the addressed rows of a validated rank-2 table, flat.
fn gathered_rows(table: &DenseArray, idx: &[u64]) -> Result<Vec<f64>, EvalError> {
    let (rows, dim) = (table.shape().dims()[0], table.shape().dims()[1]);
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &row in idx {
        let row = row as usize;
        if row >= rows {
            return Err(EvalError::Unsupported(format!(
                "gather_rows: index {row} out of range for {rows} rows"
            )));
        }
        out.extend_from_slice(&table.data()[row * dim..(row + 1) * dim]);
    }
    Ok(out)
}
