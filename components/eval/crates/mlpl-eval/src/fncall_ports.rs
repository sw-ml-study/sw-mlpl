//! The general Port builtins: `port_send` / `port_poll` / `port_recv`
//! over an opaque port handle (a `Value::ExtHandle`). Share-nothing
//! message passing with a far-end service (a provider UI thread, or a
//! test echo worker): only owned `Value`s cross the channel, so the
//! interpreter and the far end cannot race. `port_send` pushes a
//! command; `port_recv` blocks for one event; `port_poll` non-blocking
//! drains the queued events into a batch record `{count, items}`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use crate::port_util::{arity_err, event_batch};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "port_send" => Some(eval_send(args, env, trace)),
        "port_recv" => Some(eval_recv(args, env, trace)),
        "port_poll" => Some(eval_poll(args, env, trace)),
        _ => None,
    }
}

/// `port_send(port, value)` -- push a command `Value` to the far end;
/// returns `1`. Errors if the far end has hung up.
fn eval_send(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(arity_err("port_send", 2, args.len()));
    }
    let handle = eval_expr(&args[0], env, trace)?;
    let command = eval_expr(&args[1], env, trace)?;
    let port = env.resolve_port(&handle)?;
    port.commands
        .send(command)
        .map_err(|_| EvalError::ExtensionError {
            function: "port_send".into(),
            message: "the far end is gone".into(),
        })?;
    Ok(Value::Array(DenseArray::from_scalar(1.0)))
}

/// `port_recv(port)` -- block until one event arrives from the far end.
fn eval_recv(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(arity_err("port_recv", 1, args.len()));
    }
    let handle = eval_expr(&args[0], env, trace)?;
    let port = env.resolve_port(&handle)?;
    let rx = port.events.lock().expect("port receiver lock");
    rx.recv().map_err(|_| EvalError::ExtensionError {
        function: "port_recv".into(),
        message: "the far end is gone".into(),
    })
}

/// `port_poll(port [, limit])` -- non-blocking drain into a batch
/// record `{count, items}`. With `limit` it returns at most that many
/// events (bounded delivery); without, it drains all queued events. An
/// empty queue yields `{count: 0, items: {}}`.
fn eval_poll(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err("port_poll", 1, args.len()));
    }
    let handle = eval_expr(&args[0], env, trace)?;
    let limit = match args.get(1) {
        Some(a) => eval_expr(a, env, trace)?.as_array()?.data()[0].max(0.0) as usize,
        None => usize::MAX,
    };
    let port = env.resolve_port(&handle)?;
    let rx = port.events.lock().expect("port receiver lock");
    Ok(event_batch(&rx, limit))
}
