//! `for <binding> in <source> { body }` evaluator (Saga 12 step 003).
//!
//! Kept in its own module so `eval.rs` stays under the sw-checklist
//! function-count and file-LOC budgets.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;

/// Evaluate a `for` loop. See the `Expr::For` variant docstring for
/// semantics. Returns the final iteration's value; also populates
/// `last_rows` in `env` with every captured per-iteration value
/// stacked along a new outer axis.
pub(crate) fn eval_for(
    binding: &str,
    source: &Expr,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let src_arr = eval_expr(source, env, trace)?.into_array()?;
    let dims = src_arr.shape().dims();
    if dims.is_empty() {
        return Err(EvalError::Unsupported(
            "for: source must have rank >= 1".into(),
        ));
    }
    let n = dims[0];
    let row_stride: usize = dims[1..].iter().product::<usize>().max(1);
    let slice_dims: Vec<usize> = dims[1..].to_vec();
    let src_labels = src_arr.labels().map(<[_]>::to_vec);
    let mut captured: Vec<DenseArray> = Vec::with_capacity(n);
    let mut last = DenseArray::from_scalar(0.0);
    for i in 0..n {
        let slice_data = src_arr.data()[i * row_stride..(i + 1) * row_stride].to_vec();
        let mut row = DenseArray::new(Shape::new(slice_dims.clone()), slice_data)?;
        if let Some(src_lbls) = src_labels.as_ref() {
            row = row.with_labels(src_lbls[1..].to_vec())?;
        }
        env.set(binding.to_string(), row);
        let mut iter_val = DenseArray::from_scalar(0.0);
        for stmt in body {
            iter_val = eval_expr(stmt, env, trace)?.into_array()?;
        }
        captured.push(iter_val.clone());
        last = iter_val;
    }
    env.set("last_rows".into(), stack_rows(&captured)?);
    Ok(("for", vec![], last))
}

/// Stack a list of captured per-iteration values into one array,
/// prepending a new outer axis of length `rows.len()`. Returns an
/// empty rank-1 array when there are no iterations. Every captured
/// value must share the same shape.
fn stack_rows(rows: &[DenseArray]) -> Result<DenseArray, EvalError> {
    if rows.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }
    let inner_dims = rows[0].shape().dims().to_vec();
    let row_elems: usize = inner_dims.iter().product::<usize>().max(1);
    let mut data = Vec::with_capacity(rows.len() * row_elems);
    for r in rows {
        if r.shape().dims() != inner_dims.as_slice() {
            return Err(EvalError::Unsupported(
                "for: captured values must all share the same shape".into(),
            ));
        }
        if r.rank() == 0 {
            data.push(r.data()[0]);
        } else {
            data.extend_from_slice(r.data());
        }
    }
    let mut out_dims = vec![rows.len()];
    out_dims.extend_from_slice(&inner_dims);
    Ok(DenseArray::new(Shape::new(out_dims), data)?)
}
