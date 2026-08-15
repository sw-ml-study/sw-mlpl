//! Step 3 of extensions-event-loop: the handler registry + dispatch
//! loop (the JS-applet model). MLPL registers handlers with `on`, then
//! `run` folds a scripted event stream through them until a `close`
//! event; `off` unregisters. Headless -- the far end is an in-process
//! sender injecting event records; no winit, no provider.

use std::collections::BTreeMap;
use std::sync::mpsc;

use mlpl_eval::{Environment, Value};

fn eval(env: &mut Environment, src: &str) -> Result<Value, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&toks).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

/// A scripted event record `{kind: <kind>}`.
fn event(kind: &str) -> Value {
    Value::Record {
        fields: BTreeMap::from([("kind".to_string(), Value::Str(kind.to_string()))]),
    }
}

#[test]
fn run_folds_events_through_handlers_off_and_close() {
    let (cmd_tx, _cmd_rx) = mpsc::channel::<Value>();
    let (ev_tx, ev_rx) = mpsc::channel::<Value>();
    let mut env = Environment::new();
    let handle = env.register_port(cmd_tx, ev_rx);
    env.ext_handles.insert("p".to_string(), handle);

    // A handler that folds +1 per "inc" event, threading state.
    eval(&mut env, "def u:on_inc(e, s) { s + 1 }").unwrap();
    eval(&mut env, "on(p, \"inc\", :u:on_inc)").unwrap();

    // Scripted stream: an unregistered "noop" is skipped; "close" stops.
    ev_tx.send(event("inc")).unwrap();
    ev_tx.send(event("noop")).unwrap(); // no handler -> skipped
    ev_tx.send(event("inc")).unwrap();
    ev_tx.send(event("close")).unwrap();
    assert_eq!(scalar(&mut env, "run(p, 0)"), 2.0); // two inc folded

    // off: unregister, then a fresh stream leaves state untouched.
    eval(&mut env, "off(p, \"inc\")").unwrap();
    ev_tx.send(event("inc")).unwrap();
    ev_tx.send(event("close")).unwrap();
    assert_eq!(scalar(&mut env, "run(p, 5)"), 5.0); // handler gone
    let _keep = ev_tx;
}
