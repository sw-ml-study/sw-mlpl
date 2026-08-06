//! The typed test-event API (mlplunit's native-test-events
//! contract): `test_event_sink(:u:f)` registers the delivery
//! callback and `emit_test_event(record)` validates the event
//! ENVELOPE loudly, then delivers the record to the sink.
//! Counting, TAP mapping, durations, and output capture are the
//! RUNNER's business -- unknown additive fields pass through
//! untouched.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// The documented event kinds (schema version 1).
pub const EVENT_KINDS: &[&str] = &[
    "suite_start",
    "test_start",
    "case_start",
    "assertion_failure",
    "output",
    "test_end",
    "case_end",
    "teardown_failure",
    "suite_end",
];

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "test_event_sink" => Some(eval_sink(args, env, trace)),
        "emit_test_event" => Some(eval_emit(args, env, trace)),
        _ => None,
    }
}

/// `test_event_sink(:u:f)` -- register the one callback for this
/// evaluation. Sinks are user code; builtin refs are rejected.
fn eval_sink(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [f] = args else {
        return Err(EvalError::BadArity {
            func: "test_event_sink".into(),
            expected: 1,
            got: args.len(),
        });
    };
    match crate::eval::eval_expr(f, env, trace)? {
        Value::UserFnRef { name } => {
            env.test_event_sink = Some(name);
            Ok(ok_one())
        }
        other => Err(EvalError::Unsupported(format!(
            "test_event_sink: the sink must be a `:u:name` user-function reference -- got {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}

/// `emit_test_event(record)` -- validate, deliver, report. The
/// sink's failure (returned err or captured hard error) becomes
/// emit's failure; with no sink registered, emit validates only.
fn eval_emit(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [event] = args else {
        return Err(EvalError::BadArity {
            func: "emit_test_event".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let event = crate::eval::eval_expr(event, env, trace)?;
    let Value::Record { ref fields } = event else {
        return Err(EvalError::Unsupported(format!(
            "emit_test_event: the event must be a record -- got {}",
            mlpl_eval_types::value_kind(&event)
        )));
    };
    crate::event_envelope::validate(fields)?;
    let Some(sink) = env.test_event_sink.clone() else {
        return Ok(ok_one());
    };
    match crate::eval_user_fn::invoke_user_fn_values(&sink, &[event], env, trace)? {
        failed @ Value::Result { ok: false, .. } => Ok(failed),
        _ => Ok(ok_one()),
    }
}

fn ok_one() -> Value {
    Value::Result {
        ok: true,
        payload: Box::new(Value::Array(DenseArray::from_scalar(1.0))),
    }
}
