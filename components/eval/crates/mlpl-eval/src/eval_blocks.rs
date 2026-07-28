//! Big-shape constructor helpers driving the tail of
//! `eval::eval_expr`: `tensor`/`param` constructors, `repeat { }`,
//! and `train { }`. Each returns the `(op_name, inputs, result)`
//! triple the trace-event emit at the end of `eval_expr` consumes.
//!
//! Saga 33 step 023 split these out of `eval.rs` so that file
//! falls below the 350-line File-LOC warning floor.

use crate::env_api::*;
use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::{Expr, TensorCtorKind};
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;

pub(crate) fn eval_tensor_ctor(
    kind: TensorCtorKind,
    shape: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let mut dims = Vec::with_capacity(shape.len());
    for dim_expr in shape {
        let arr = eval_expr(dim_expr, env, trace)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::InvalidShapeDim);
        }
        let v = arr.data()[0];
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EvalError::InvalidShapeDim);
        }
        dims.push(v as usize);
    }
    let zeros = DenseArray::zeros(Shape::new(dims));
    // Construct an autograd Tensor on a fresh tape. Step 005 will
    // route this through a tape stored in the environment so that
    // operations on the resulting array are recorded; for now we
    // simply return the underlying zero-initialized array.
    let tape = Tape::new();
    let _tensor = match kind {
        TensorCtorKind::Param => Tensor::param(tape, zeros.clone()),
        TensorCtorKind::Tensor => Tensor::leaf(tape, zeros.clone(), false),
    };
    let op_name = match kind {
        TensorCtorKind::Param => "param_ctor",
        TensorCtorKind::Tensor => "tensor_ctor",
    };
    Ok((op_name, vec![], zeros))
}

pub(crate) fn eval_repeat(
    count: &Expr,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let n_arr = eval_expr(count, env, trace)?.into_array()?;
    if n_arr.rank() != 0 {
        return Err(EvalError::InvalidRepeatCount);
    }
    let n = n_arr.data()[0] as usize;
    let mut r = DenseArray::from_scalar(0.0);
    for i in 0..n {
        // Saga 21.5 step 003: cancellation checkpoint at each
        // iteration head. The bubbled error carries the iteration
        // index so clients can show "cancelled after N reps".
        if env.check_interrupt().is_err() {
            return Err(EvalError::Cancelled {
                step: i,
                partial_losses: Vec::new(),
            });
        }
        for stmt in body {
            r = eval_expr(stmt, env, trace)?.into_array()?;
        }
    }
    Ok(("repeat", vec![], r))
}

pub(crate) fn eval_train(
    count: &Expr,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let n_arr = eval_expr(count, env, trace)?.into_array()?;
    if n_arr.rank() != 0 {
        return Err(EvalError::InvalidRepeatCount);
    }
    let n = n_arr.data()[0] as usize;
    let mut losses: Vec<f64> = Vec::with_capacity(n);
    let mut last = DenseArray::from_scalar(0.0);
    for i in 0..n {
        // Saga 21.5 step 003: cancellation checkpoint at iter head.
        if env.check_interrupt().is_err() {
            return crate::interrupt::enrich_train_cancel(env, i, losses);
        }
        env.set("step".into(), DenseArray::from_scalar(i as f64));
        let mut step_val = DenseArray::from_scalar(0.0);
        for stmt in body {
            step_val = match eval_expr(stmt, env, trace) {
                Ok(v) => v.into_array()?,
                Err(EvalError::Cancelled { .. }) => {
                    return crate::interrupt::enrich_train_cancel(env, i, losses);
                }
                Err(e) => return Err(e),
            };
        }
        // Body's final value is the per-step loss; non-scalar
        // values reduce by mean so callers always get a scalar
        // history.
        let scalar_loss = if step_val.rank() == 0 {
            step_val.data()[0]
        } else {
            let s: f64 = step_val.data().iter().sum();
            s / (step_val.data().len().max(1) as f64)
        };
        losses.push(scalar_loss);
        last = step_val;
        env.emit_metrics(i, scalar_loss);
    }
    let losses_arr =
        DenseArray::new(Shape::new(vec![losses.len()]), losses).expect("losses shape matches data");
    env.set("last_losses".into(), losses_arr);
    Ok(("train", vec![], last))
}
