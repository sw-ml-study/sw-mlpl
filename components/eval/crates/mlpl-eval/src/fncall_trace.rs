//! Shared trace-event push helper for FnCall dispatchers that
//! emit a single array-valued trace event.
//!
//! Saga 33 step 023 lifted the `if let Some(t) = trace.as_mut()`
//! boilerplate out of `matmul` / `momentum_sgd` / `adam` so each
//! caller stays under the 25-LOC function gate.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use mlpl_core::Span;
use mlpl_eval_types::{EvalError, Value};
use mlpl_trace::{Trace, TraceEvent, TraceValue};

pub(crate) fn push_array_event(
    trace: &mut Option<&mut Trace>,
    op: &str,
    span: &Span,
    inputs: Vec<TraceValue>,
    result: &DenseArray,
) {
    if let Some(t) = trace.as_mut() {
        let seq = t.events().len() as u64;
        t.push(TraceEvent {
            seq,
            op: op.into(),
            span: *span,
            inputs,
            output: TraceValue::from_array(result),
            input_types: vec![],
            output_type: None,
        });
    }
}

/// `emit_frame(name, step, x)` -- push tensor `x` as a live FRAME
/// through the installed `MetricSink` (the `emit` analog for whole
/// boards/images; Game of Life saga step 4). No sink installed --
/// the local browser/REPL case -- is a no-op. Returns `x`, so the
/// call composes inside loop bodies.
pub(crate) fn eval_emit_frame(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "emit_frame".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let Expr::StrLit(name, _) = &args[0] else {
        return Err(EvalError::Unsupported(
            "emit_frame: name must be a string literal".into(),
        ));
    };
    let step_arr = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    let x = crate::eval::eval_expr(&args[2], env, trace)?.into_array()?;
    if let Some(sink) = env.metric_sink() {
        let step = step_arr.data().first().copied().unwrap_or(0.0) as usize;
        sink.emit_frame(name, step, x.shape().dims(), x.data());
    }
    Ok(Value::Array(x))
}
