//! Shared trace-event push helper for FnCall dispatchers that
//! emit a single array-valued trace event.
//!
//! Saga 33 step 023 lifted the `if let Some(t) = trace.as_mut()`
//! boilerplate out of `matmul` / `momentum_sgd` / `adam` so each
//! caller stays under the 25-LOC function gate.

use mlpl_array::DenseArray;
use mlpl_core::Span;
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
