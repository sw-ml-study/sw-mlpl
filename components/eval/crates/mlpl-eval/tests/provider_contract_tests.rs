//! Step 5 of extensions-event-loop: the provider contract, proven
//! headlessly. A provider (demo-extensions wgpu/winit) plugs in as the
//! UI host -- the `FnOnce(cmd_rx, ev_tx)` run on the main thread by
//! `run_applet_with_host`. It forwards typed, ordered, bounded event
//! records (`{kind, ...}`) into the event channel and drains the
//! commands the applet submits. Here the host is an in-process double
//! (no winit); the same contract a real provider implements.

use std::collections::BTreeMap;

use mlpl_eval::{Value, run_applet_with_host};

/// A scripted event record `{kind: <kind>}`, as a provider would emit.
fn event(kind: &str) -> Value {
    Value::Record {
        fields: BTreeMap::from([("kind".to_string(), Value::Str(kind.to_string()))]),
    }
}

#[test]
fn a_provider_host_drives_typed_handlers_and_receives_commands() {
    // Applet: one handler per event kind, each submitting a tagged
    // render command and folding a distinct amount into the state.
    let source = "def u:on_key(e, s) {\n  port_send(port, 1)\n  s + 1\n}\n\
                  def u:on_ptr(e, s) {\n  port_send(port, 2)\n  s + 10\n}\n\
                  on(port, \"key\", :u:on_key)\n\
                  on(port, \"pointer\", :u:on_ptr)\n\
                  run(port, 0)";

    // The provider-shaped host (runs on this thread): an ordered, typed
    // event stream ending in close, draining the commands submitted.
    let mut commands: Vec<f64> = Vec::new();
    let result = run_applet_with_host(source, |cmd_rx, ev_tx| {
        for kind in ["key", "pointer", "key", "close"] {
            ev_tx.send(event(kind)).unwrap();
        }
        for _ in 0..3 {
            if let Ok(Value::Array(a)) = cmd_rx.recv() {
                commands.push(a.data()[0]);
            }
        }
    })
    .unwrap();

    // key + pointer + key folds to 1 + 10 + 1 = 12, quitting on close.
    match result {
        Value::Array(a) => assert_eq!(a.data()[0], 12.0),
        other => panic!("expected scalar result, got {other:?}"),
    }
    // Per-kind commands arrive in event order.
    assert_eq!(commands, vec![1.0, 2.0, 1.0]);
}
