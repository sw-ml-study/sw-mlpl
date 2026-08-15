//! The event-dispatch builtins -- the JS-applet model over a Port:
//! `on(port, event, :u:handler)` / `off(port, event)` register and
//! unregister handlers, and `run(port, state)` is the dispatch loop
//! that pulls events, invokes the matching handler in-process, and
//! FOLDS app state as a value (`state = handler(event, state)`) until a
//! `close` event -- returning the final state. Runs on the worker
//! thread; the far end (a provider UI thread or a test) only forwards
//! event records with a `kind` field.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use crate::eval_user_fn::invoke_user_fn_values;
use crate::port_util::{arity_err, event_kind, kind_err, port_slot};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "on" => Some(eval_on_off(args, env, trace, true)),
        "off" => Some(eval_on_off(args, env, trace, false)),
        "run" => Some(eval_run(args, env, trace)),
        _ => None,
    }
}

/// `on(port, "event", :u:handler)` registers a handler; `off(port,
/// "event")` unregisters. The handler key is the `u:`-prefixed fn name.
fn eval_on_off(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    register: bool,
) -> Result<Value, EvalError> {
    let expected = if register { 3 } else { 2 };
    if args.len() != expected {
        return Err(arity_err("on/off", expected, args.len()));
    }
    let handle = eval_expr(&args[0], env, trace)?;
    let slot = port_slot(&handle)?;
    let Value::Str(event) = eval_expr(&args[1], env, trace)? else {
        return Err(kind_err("event name must be a string"));
    };
    if !register {
        env.port_handlers.remove(&(slot, event));
        return Ok(Value::Array(DenseArray::from_scalar(1.0)));
    }
    let Value::UserFnRef { name } = eval_expr(&args[2], env, trace)? else {
        return Err(kind_err("handler must be a :u: function ref"));
    };
    env.port_handlers.insert((slot, event), name);
    Ok(Value::Array(DenseArray::from_scalar(1.0)))
}

/// `run(port, state)` -- the dispatch loop. Blocks pulling the next
/// event, folds it through the matching handler, and stops on a `close`
/// event, returning the final state.
fn eval_run(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(arity_err("run", 2, args.len()));
    }
    let handle = eval_expr(&args[0], env, trace)?;
    let slot = port_slot(&handle)?;
    let mut state = eval_expr(&args[1], env, trace)?;
    // A dropped far end (no close sent) ends the while-let naturally.
    while let Ok(event) = env.resolve_port(&handle)?.events.recv() {
        let (keep_going, next) = dispatch_one(env, slot, event, state, trace)?;
        state = next;
        if !keep_going {
            break;
        }
    }
    Ok(state)
}

/// Fold one event: `close` stops the loop (returns `(false, state)`);
/// otherwise the matching handler (if any) folds it into the state.
fn dispatch_one(
    env: &mut Environment,
    slot: u32,
    event: Value,
    state: Value,
    trace: &mut Option<&mut Trace>,
) -> Result<(bool, Value), EvalError> {
    let kind = event_kind(&event);
    if kind == "close" {
        return Ok((false, state));
    }
    match env.port_handlers.get(&(slot, kind)).cloned() {
        Some(f) => Ok((
            true,
            invoke_user_fn_values(&f, &[event, state], env, trace)?,
        )),
        None => Ok((true, state)),
    }
}
