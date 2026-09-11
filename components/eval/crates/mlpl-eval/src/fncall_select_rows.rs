//! `select_rows(table, keys)` -- vectorized keyed row lookup.
//!
//! `table` is a record mapping each key to a numeric ROW (a rank-1 array of
//! a common width C); `keys` is a string list of N keys. The result is the
//! `[N, C]` matrix whose row i is `table[keys[i]]`. It turns a categorical
//! list into an attribute matrix in one call -- e.g. a region-kind list into
//! an `[N,4]` RGBA selection, or any label -> row mapping -- which is
//! otherwise a verbose manual loop (and impossible today, since `take` does
//! not row-select).

use std::collections::BTreeMap;

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, value_kind};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "select_rows" => Some(eval_select_rows(args, env, trace)),
        _ => None,
    }
}

fn eval_select_rows(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [table_arg, keys_arg] = args else {
        return Err(EvalError::BadArity {
            func: "select_rows".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let table = match eval_expr(table_arg, env, trace)? {
        Value::Record { fields } => fields,
        v => return Err(want("first argument must be a record of key -> row", &v)),
    };
    let keys = match eval_expr(keys_arg, env, trace)? {
        Value::StrList { items } => items,
        v => return Err(want("second argument must be a string list of keys", &v)),
    };
    build_matrix(&table, &keys)
}

/// Gather `table[key]` for each key into an `[N, C]` matrix.
fn build_matrix(table: &BTreeMap<String, Value>, keys: &[String]) -> Result<Value, EvalError> {
    if keys.is_empty() {
        return Err(EvalError::Unsupported(
            "select_rows: keys must be non-empty (no width to infer otherwise)".into(),
        ));
    }
    let mut width: Option<usize> = None;
    let mut data: Vec<f64> = Vec::new();
    for key in keys {
        let row = lookup_row(table, key)?;
        if *width.get_or_insert(row.len()) != row.len() {
            return Err(EvalError::Unsupported(format!(
                "select_rows: row \"{key}\" has {} elements, expected {}",
                row.len(),
                width.unwrap()
            )));
        }
        data.extend_from_slice(row);
    }
    let shape = Shape::new(vec![keys.len(), width.unwrap_or(0)]);
    Ok(Value::Array(DenseArray::new(shape, data)?))
}

/// Fetch `table[key]` as a flat numeric row, or a clear error.
fn lookup_row<'a>(table: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a [f64], EvalError> {
    match table.get(key) {
        Some(Value::Array(a)) => Ok(a.data()),
        Some(v) => Err(want(
            &format!("value for key \"{key}\" must be a numeric row"),
            v,
        )),
        None => {
            let keys: Vec<&str> = table.keys().map(String::as_str).collect();
            Err(EvalError::Unsupported(format!(
                "select_rows: no row for key \"{key}\" (available: {})",
                keys.join(", ")
            )))
        }
    }
}

fn want(reason: &str, got: &Value) -> EvalError {
    EvalError::Unsupported(format!("select_rows: {reason}, got a {}", value_kind(got)))
}
