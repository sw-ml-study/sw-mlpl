//! Hook invocation and result merging for `bracket` -- the
//! error plane of docs/finally-design.md: hard evaluator errors
//! are captured at each hook boundary as the same structured
//! `{kind, message}` record `try`/`catch` produces, and when
//! use AND teardown both fail the teardown diagnostic is
//! retained under `teardown_error` while use's failure stays
//! primary.

use std::collections::BTreeMap;

use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Invoke one hook, capturing hard errors as `err({kind,
/// message})`. Control signals (`break`/`continue`/`return`)
/// pass through untouched.
pub(crate) fn invoke(
    callable: &Value,
    fixture: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    match crate::callable_apply::apply_callable(callable, fixture, env, trace) {
        Ok(v) => Ok(v),
        Err(
            sig @ (EvalError::BreakSignal(_)
            | EvalError::ContinueSignal
            | EvalError::ReturnSignal(_)
            | EvalError::ExitRequested(_)),
        ) => Err(sig),
        Err(e) => {
            let fields = BTreeMap::from([
                (
                    "kind".to_string(),
                    Value::Str(mlpl_eval_types::error_kind(&e).to_string()),
                ),
                ("message".to_string(), Value::Str(format!("{e}"))),
            ]);
            Ok(Value::Result {
                ok: false,
                payload: Box::new(Value::Record { fields }),
            })
        }
    }
}

/// Result precedence: use's failure is PRIMARY; a teardown
/// failure after a successful use is a real failure (leaked
/// resource); when both fail, the teardown diagnostic rides
/// along as `teardown_error`.
pub(crate) fn merge(used: Value, cleaned: Value) -> Value {
    match (used, cleaned) {
        (
            Value::Result {
                ok: false,
                payload: primary,
            },
            Value::Result {
                ok: false,
                payload: teardown,
            },
        ) => Value::Result {
            ok: false,
            payload: Box::new(attach(*primary, *teardown)),
        },
        (primary @ Value::Result { ok: false, .. }, _) => primary,
        (_, failed @ Value::Result { ok: false, .. }) => failed,
        (primary, _) => primary,
    }
}

/// A record primary gains the `teardown_error` field; any other
/// payload wraps to `{message, teardown_error}`.
fn attach(primary: Value, teardown: Value) -> Value {
    let mut fields = match primary {
        Value::Record { fields } => fields,
        other => BTreeMap::from([("message".to_string(), other)]),
    };
    fields.insert("teardown_error".to_string(), teardown);
    Value::Record { fields }
}
