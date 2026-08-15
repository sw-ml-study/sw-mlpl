//! Step 2 of extensions-event-loop: the general Port primitive. MLPL
//! sends commands to and receives events from a far-end service over
//! channels, share-nothing -- only owned `Value`s cross, so the
//! interpreter and the far end cannot race. The far end here is an
//! in-process ECHO worker on a spawned thread (no winit, no provider).

use std::sync::mpsc;
use std::thread;

use mlpl_array::DenseArray;
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

#[test]
fn commands_and_events_cross_a_thread_boundary() {
    // Echo worker on a spawned thread: recv a command, send it straight
    // back as an event. Only owned Values cross the channels.
    let (cmd_tx, cmd_rx) = mpsc::channel::<Value>();
    let (ev_tx, ev_rx) = mpsc::channel::<Value>();
    let worker = thread::spawn(move || {
        while let Ok(v) = cmd_rx.recv() {
            if ev_tx.send(v).is_err() {
                break;
            }
        }
    });
    let mut env = Environment::new();
    let handle = env.register_port(cmd_tx, ev_rx);
    env.ext_handles.insert("p".to_string(), handle);

    assert_eq!(scalar(&mut env, "port_send(p, 42)"), 1.0);
    assert_eq!(scalar(&mut env, "port_recv(p)"), 42.0); // echoed cross-thread
    drop(env); // closes the command sender -> the worker exits
    worker.join().expect("echo worker");
}

#[test]
fn poll_drains_queued_events_then_empties() {
    // The test holds the event SENDER and injects directly, so poll's
    // non-blocking drain is deterministic (no worker-timing race).
    let (cmd_tx, _cmd_rx) = mpsc::channel::<Value>();
    let (ev_tx, ev_rx) = mpsc::channel::<Value>();
    let mut env = Environment::new();
    let handle = env.register_port(cmd_tx, ev_rx);
    env.ext_handles.insert("p".to_string(), handle);

    // empty queue -> an empty batch
    assert_eq!(scalar(&mut env, "port_poll(p).count"), 0.0);
    ev_tx
        .send(Value::Array(DenseArray::from_scalar(10.0)))
        .unwrap();
    ev_tx
        .send(Value::Array(DenseArray::from_scalar(20.0)))
        .unwrap();
    assert_eq!(scalar(&mut env, "port_poll(p).count"), 2.0);
    // the drain consumed them
    assert_eq!(scalar(&mut env, "port_poll(p).count"), 0.0);
    let _keep = ev_tx;
}
