//! `dataflow(nodes, edges)` -- the structural SVG renderer builtin.
//! `nodes` is a record with a `labels` string-list (plus optional
//! `groups` id array and `highlight` 0/1 array); `edges` is a record
//! with `from` / `to` integer index arrays (plus optional `labels`
//! string-list, `widths` array, and `highlight` 0/1 array). Extracts
//! those record fields into plain slices and hands them to
//! `mlpl_viz::render_dataflow`, returning the SVG string. Interpreter /
//! web / `-f` only (a visualization surface, like `svg`); it does not
//! lower on the compile-to-Rust path.

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
    let n = record_fields("dataflow: nodes", &nodes_val)?;
    let e = record_fields("dataflow: edges", &edges_val)?;
    let labels = str_list("dataflow: nodes.labels", n.get("labels"))?;
    let groups = index_array("dataflow: nodes.groups", n.get("groups"))?;
    let node_hl = bool_array("dataflow: nodes.highlight", n.get("highlight"))?;
    let from = index_array("dataflow: edges.from", e.get("from"))?;
    let to = index_array("dataflow: edges.to", e.get("to"))?;
    let edge_labels = str_list("dataflow: edges.labels", e.get("labels"))?;
    let widths = float_array("dataflow: edges.widths", e.get("widths"))?;
    let edge_hl = bool_array("dataflow: edges.highlight", e.get("highlight"))?;
    let width_log = width_scale("dataflow: edges.width_scale", e.get("width_scale"))?;
    let scalar = |what, v| {
        Ok::<i32, EvalError>(float_array(what, v)?.first().copied().unwrap_or(0.0) as i32)
    };
    let col_gap = scalar("dataflow: nodes.col_gap", n.get("col_gap"))?;
    let row_gap = scalar("dataflow: nodes.row_gap", n.get("row_gap"))?;
    mlpl_viz::render_dataflow(&mlpl_viz::Dataflow {
        labels: &labels,
        from: &from,
        to: &to,
        edge_labels: &edge_labels,
        groups: &groups,
        node_highlight: &node_hl,
        edge_widths: &widths,
        edge_highlight: &edge_hl,
        width_log,
        col_gap,
        row_gap,
    })
    .map_err(|err| EvalError::Unsupported(format!("dataflow: {err}")))
}

/// A record value's fields, or a clear error.
fn record_fields<'a>(what: &str, v: &'a Value) -> Result<&'a BTreeMap<String, Value>, EvalError> {
    match v {
        Value::Record { fields } => Ok(fields),
        _ => Err(EvalError::Unsupported(format!("{what} must be a record"))),
    }
}

/// An optional `StrList` field: absent -> empty, present -> its items.
fn str_list(what: &str, v: Option<&Value>) -> Result<Vec<String>, EvalError> {
    match v {
        None => Ok(Vec::new()),
        Some(Value::StrList { items }) => Ok(items.clone()),
        _ => Err(EvalError::Unsupported(format!(
            "{what} must be a string list"
        ))),
    }
}

/// An optional numeric array field of non-negative integers -> indices.
fn index_array(what: &str, v: Option<&Value>) -> Result<Vec<usize>, EvalError> {
    let nums = float_array(what, v)?;
    nums.iter()
        .map(|&x| {
            if x >= 0.0 && x.fract() == 0.0 {
                Ok(x as usize)
            } else {
                Err(EvalError::Unsupported(format!(
                    "{what}: {x} is not a non-negative integer"
                )))
            }
        })
        .collect()
}

/// An optional numeric array field -> its values (absent -> empty).
fn float_array(what: &str, v: Option<&Value>) -> Result<Vec<f64>, EvalError> {
    match v {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => Ok(a.data().to_vec()),
        _ => Err(EvalError::Unsupported(format!(
            "{what} must be a numeric array"
        ))),
    }
}

/// An optional 0/1 array field -> per-element flags (nonzero = true).
fn bool_array(what: &str, v: Option<&Value>) -> Result<Vec<bool>, EvalError> {
    Ok(float_array(what, v)?.iter().map(|&x| x != 0.0).collect())
}

/// The optional `width_scale` string: absent / "linear" -> false,
/// "log" -> true, anything else -> a clean error.
fn width_scale(what: &str, v: Option<&Value>) -> Result<bool, EvalError> {
    match v {
        None => Ok(false),
        Some(Value::Str(s)) => match s.as_str() {
            "linear" => Ok(false),
            "log" => Ok(true),
            other => Err(EvalError::Unsupported(format!(
                "{what}: {other:?} must be \"linear\" or \"log\""
            ))),
        },
        _ => Err(EvalError::Unsupported(format!("{what} must be a string"))),
    }
}
