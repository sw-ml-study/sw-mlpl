//! Step 4 of extensions-event-loop: the 'parked main' launch inversion.
//! The interpreter runs an applet on a WORKER thread while a UI host on
//! the calling (main) thread drives it -- feeding events and draining
//! the commands the handlers submit. Headless: the host is a scripted
//! double, no winit.

use std::collections::BTreeMap;

use mlpl_eval::{Environment, Value, require_ui_host_thread, run_applet_with_host};

/// A scripted event record `{kind: <kind>}`.
fn event(kind: &str) -> Value {
    Value::Record {
        fields: BTreeMap::from([("kind".to_string(), Value::Str(kind.to_string()))]),
    }
}

#[test]
fn an_applet_runs_on_a_worker_driven_by_a_main_thread_host() {
    // The applet: a handler that folds +1 per "inc" and submits the new
    // count as a command; then it hands control to the dispatch loop.
    let source = "def u:on_inc(e, s) {\n  n = s + 1\n  port_send(port, n)\n  n\n}\n\
                  on(port, \"inc\", :u:on_inc)\n\
                  run(port, 0)";

    // The headless host (runs on THIS thread) scripts two inc events
    // then close, and drains the two commands the applet submitted.
    let mut commands: Vec<f64> = Vec::new();
    let result = run_applet_with_host(source, |cmd_rx, ev_tx| {
        ev_tx.send(event("inc")).unwrap();
        ev_tx.send(event("inc")).unwrap();
        ev_tx.send(event("close")).unwrap();
        for _ in 0..2 {
            if let Ok(Value::Array(a)) = cmd_rx.recv() {
                commands.push(a.data()[0]);
            }
        }
    })
    .unwrap();

    match result {
        Value::Array(a) => assert_eq!(a.data()[0], 2.0), // folded to 2, quit on close
        other => panic!("expected scalar result, got {other:?}"),
    }
    assert_eq!(commands, vec![1.0, 2.0]); // handler submitted each new count
}

#[test]
fn native_ui_requires_the_main_thread_launch_path() {
    // A default env (a connect/serve worker eval) is not UI-capable.
    let mut env = Environment::new();
    assert!(require_ui_host_thread(&env).is_err());
    // The applet launcher marks its worker env UI-capable.
    env.ui_host_thread = true;
    assert!(require_ui_host_thread(&env).is_ok());
}
