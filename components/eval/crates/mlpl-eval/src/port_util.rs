//! Small shared helpers for the port dispatch builtins (`on` / `off` /
//! `run`): extract a port slot from a handle, read an event's `kind`,
//! and build the two error kinds. Kept separate from `fncall_dispatch`
//! so each module stays within its function budget.

use crate::env::Environment;
use mlpl_eval_env::PORT_EXTENSION_ID;
use mlpl_eval_types::{EvalError, Value};

/// Block for the next event on a port, or `None` if the far end has
/// hung up. The receiver lock is taken and released within this call
/// (only the interpreter thread ever locks it, so it is uncontended),
/// so the caller's env borrow is free by the time it returns.
pub(crate) fn next_event(env: &Environment, handle: &Value) -> Result<Option<Value>, EvalError> {
    let port = env.resolve_port(handle)?;
    let rx = port.events.lock().expect("port receiver lock");
    Ok(rx.recv().ok())
}

/// The port slot from a port handle, or a boundary error if the value
/// is not a port handle.
pub(crate) fn port_slot(handle: &Value) -> Result<u32, EvalError> {
    match handle {
        Value::ExtHandle {
            extension_id, slot, ..
        } if *extension_id == PORT_EXTENSION_ID => Ok(*slot),
        other => Err(kind_err(&format!(
            "expected a port handle, got {}",
            mlpl_eval_types::value_kind(other)
        ))),
    }
}

/// An event's `kind` field as a string (`""` if the event is not a
/// record or has no string `kind`, so no handler matches).
pub(crate) fn event_kind(event: &Value) -> String {
    match event {
        Value::Record { fields } => match fields.get("kind") {
            Some(Value::Str(s)) => s.clone(),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// A `BadArity` error for a port builtin.
pub(crate) fn arity_err(func: &str, expected: usize, got: usize) -> EvalError {
    EvalError::BadArity {
        func: func.into(),
        expected,
        got,
    }
}

/// A boundary error for a port builtin.
pub(crate) fn kind_err(message: &str) -> EvalError {
    EvalError::ExtensionError {
        function: "port".into(),
        message: message.into(),
    }
}
