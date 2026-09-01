//! `dataflow(nodes, edges)` -- the structural SVG renderer builtin.
//! `nodes` is a record with a `labels` string-list; `edges` is a
//! record with `from` / `to` integer index arrays and an optional
//! `labels` string-list. Extracts those record fields into plain
//! slices and hands them to `mlpl_viz::render_dataflow`, returning the
//! SVG string. Interpreter / web / `-f` only (a visualization surface,
//! like `svg`); it does not lower on the compile-to-Rust path.

use std::collections::BTreeMap;

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

/// `dataflow(nodes, edges)` -> SVG string.
pub(crate) fn eval_dataflow(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "dataflow".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let nodes_val = eval_expr(&args[0], env, trace)?;
    let edges_val = eval_expr(&args[1], env, trace)?;
    let nodes = record_fields("dataflow: nodes", &nodes_val)?;
    let edges = record_fields("dataflow: edges", &edges_val)?;
    let labels = str_list("dataflow: nodes.labels", nodes.get("labels"))?;
    let from = index_array("dataflow: edges.from", edges.get("from"))?;
    let to = index_array("dataflow: edges.to", edges.get("to"))?;
    let edge_labels = match edges.get("labels") {
        Some(_) => str_list("dataflow: edges.labels", edges.get("labels"))?,
        None => Vec::new(),
    };
    mlpl_viz::render_dataflow(&labels, &from, &to, &edge_labels)
        .map_err(|e| EvalError::Unsupported(format!("dataflow: {e}")))
}

/// A record value's fields, or a clear error.
fn record_fields<'a>(what: &str, v: &'a Value) -> Result<&'a BTreeMap<String, Value>, EvalError> {
    match v {
        Value::Record { fields } => Ok(fields),
        _ => Err(EvalError::Unsupported(format!("{what} must be a record"))),
    }
}

/// A record field that must be a `StrList`, cloned to a `Vec<String>`.
fn str_list(what: &str, v: Option<&Value>) -> Result<Vec<String>, EvalError> {
    match v {
        Some(Value::StrList { items }) => Ok(items.clone()),
        _ => Err(EvalError::Unsupported(format!(
            "{what} must be a string list"
        ))),
    }
}

/// A record field that must be a numeric array of non-negative
/// integers, converted to `Vec<usize>` node ids.
fn index_array(what: &str, v: Option<&Value>) -> Result<Vec<usize>, EvalError> {
    match v {
        Some(Value::Array(a)) => a
            .data()
            .iter()
            .map(|&x| {
                if x >= 0.0 && x.fract() == 0.0 {
                    Ok(x as usize)
                } else {
                    Err(EvalError::Unsupported(format!(
                        "{what}: {x} is not a non-negative integer"
                    )))
                }
            })
            .collect(),
        _ => Err(EvalError::Unsupported(format!(
            "{what} must be an integer array"
        ))),
    }
}
